import gc
import math
import os
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    GenerationConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, PrinterCallback
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    prepare_deepspeed,
)
from peft import PeftModel
from .trainer_config import TrainerConfig
from ..utils.eval_utils import eval_func_gsm8k, eval_func_arc
from ..utils.trainer_utils import (
    OnlineTrainerState,
    batch_generation,
    truncate_response,
    pad_mask_rearrange,
    get_logprob_reward
)

class BaseTrainer(Trainer):
    """
    Args:
        config (`TrainerConfig`):
            training configs,
        tokenizer (`PreTrainedTokenizer`):
            tokenizer for the models, assuming all models are from the same family
        policy (`nn.Module`):
            the LM "q_phi" to sample rationale from
        ref_policy (`nn.Module`):
            the reference model "p_0"
        train_dataset (`Dataset`):
            training dataset, should have columns "queries" and "responses"
        eval_dataset (`Dataset`):
            evaluation dataset used during training epochs, should have columns "queries" and "responses"
        optimizers (`torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`):
            a tuple of (optimizer, learning rate scheduler) for optimization,
            for now always prepare an optimizer for all models when init the trainer.
        callbacks (`Optional[List[TrainerCallback]]`):
            callback functions used to customize the training
    """
    def __init__(
        self,
        config: TrainerConfig,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        policy: nn.Module,
        ref_policy: Optional[nn.Module] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        # args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * accelerator.num_processes)
        args.batch_size = int(args.local_batch_size * accelerator.num_processes)
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_evaluations > 0:
            self.evaluation_freq = max(1, args.num_total_batches // args.num_evaluations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy]:
            disable_dropout_in_model(module)
        self.model = policy
        if self.optimizer is None:
            # NOTE: this default function creates an optimizer only for `self.model`
            # It's okay to not pass an optimizer when the reward_model to train is the same as self.model
            # But when different, we need to have two optimizers for each sub-loop during training.
            self.create_optimizer_and_scheduler(
                num_training_steps=args.num_total_batches
            )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
        )
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model # NOTE: internal use
            if not isinstance(policy, PeftModel):
                # NOTE: when policy is PeftModel, ref_policy will be the base model therein.
                # prepare this model twice will lead to error
                self.ref_policy = prepare_deepspeed(
                    self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)


    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def train(self):
        pass

    def log_training_samples(self, queries, responses, postprocessed_responses, query_ppresponse_finals, final_responses, rewards, advantages):
        table = defaultdict(list)
        table["query"] = gather_object(self.tokenizer.batch_decode(queries))
        table["model_response"] = gather_object(self.tokenizer.batch_decode(responses))
        table["postprocessed_response"] = gather_object(self.tokenizer.batch_decode(postprocessed_responses))
        table["assembled_sequences"] = gather_object(self.tokenizer.batch_decode(query_ppresponse_finals))
        table["groundtruth"] = gather_object(self.tokenizer.batch_decode(final_responses))
        table["reward"] = self.accelerator.gather(rewards).float().cpu().tolist()
        table["advantage"] = self.accelerator.gather(advantages).float().cpu().tolist()
        df = pd.DataFrame(table)
        if wandb.run is not None:
            wandb.log({"training_samples": wandb.Table(dataframe=df)})

    def evaluation(self):
        tokenizer = self.tokenizer
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.0 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model, torch.no_grad():
            for batch in self.eval_dataloader:
                query = tokenizer(batch["queries"], padding=True, return_tensors="pt")["input_ids"].to(self.accelerator.device)
                final_response = tokenizer(batch["responses"], padding=True, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.accelerator.device)
                context_length = query.shape[1]
                query_response, _ = batch_generation(
                    unwrapped_model,
                    query,
                    query.shape[0],
                    tokenizer.pad_token_id,
                    generation_config,
                )
                del _
                torch.cuda.empty_cache()
                response = query_response[:, context_length:]
                postprocessed_response, _ = truncate_response([tokenizer.eos_token_id, tokenizer.pad_token_id], tokenizer.pad_token_id, response)
                table["query"].extend(gather_object(tokenizer.batch_decode(query, skip_special_tokens=True)))
                table["raw_response"].extend(gather_object(tokenizer.batch_decode(response)))
                table["model_response"].extend(gather_object(tokenizer.batch_decode(postprocessed_response, skip_special_tokens=True)))
                table["final_response"].extend(gather_object(tokenizer.batch_decode(final_response, skip_special_tokens=True)))
                table["groundtruth"].extend(gather_object(batch["groundtruth"]))
                # compute rewards
                query_ppresponse = torch.cat((query, postprocessed_response), 1)
                query_ppresponse_final = torch.cat((query_ppresponse, final_response), 1)
                query_ppresponse_final = pad_mask_rearrange(query_ppresponse_final, tokenizer.pad_token_id)
                reward_context_length = (query_ppresponse.shape[1] + first_true_indices(final_response != tokenizer.pad_token_id) - 1).unsqueeze(1)

                # assuming reward_model is self.model
                reward = get_logprob_reward(unwrapped_model, tokenizer.pad_token_id, query_ppresponse_final, reward_context_length)
                table["reward"].extend(self.accelerator.gather(reward.squeeze(-1)).float().cpu().tolist())
                del response, postprocessed_response, final_response, query_ppresponse, query_ppresponse_final, reward_context_length, reward
                torch.cuda.empty_cache()
                gc.collect()
        df = pd.DataFrame(table)
        df = df[:len(self.eval_dataset)]
        match self.args.dataset_name:
            case "gsm8k":
                eval_func = eval_func_gsm8k
            case "arc_challenge":
                eval_func = eval_func_arc
            case _:
                raise ValueError(f"Unknown dataset {self.args.dataset_name}")
        df["result"] = df.apply(eval_func, axis=1)
        acc = df["result"].mean()
        if acc > self.state.best_metric:
            self.state.best_metric = acc
            self.accelerator.print("====Saving checkpoint====")
            self._save_checkpoint(self.model, trial=None, metrics=None)

        self.log({"objective/eval_reward": df["reward"].mean(), "eval/zero_shot_acc": acc})

        if wandb.run is not None:
            wandb.log({"evaluations": wandb.Table(dataframe=df)})
