import gc
import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    GenerationConfig,
    PreTrainedTokenizer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import deepspeed_load_checkpoint
from transformers.utils import is_sagemaker_mp_enabled
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    first_true_indices,
)
from .trainer_config import TrainerConfig
from .base_trainer import BaseTrainer
from ..utils.trainer_utils import (
    generate,
    truncate_response,
    pad_mask_rearrange,
    get_logprob_reward,
)

class LatroTrainer(BaseTrainer):
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
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            policy=policy,
            ref_policy=ref_policy,
            eval_dataset=eval_dataset,
            optimizers=optimizers,
            callbacks=callbacks
        )

    def train(self, resume_from_checkpoint: str = None):
        args = self.args
        accelerator = self.accelerator
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = accelerator.device
        def repeat_generator():
            while True:
                yield from dataloader
        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=True # not LoRA
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        model = self.model
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        optimizer = self.optimizer

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=1, # (args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        rloo_loss_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
        sft_loss_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
        model.train()

        # trainer state initialization
        self.state.best_metric = 0.0
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            # Part 1: Generating responses for (local_dataloader_batch_size * rloo_k) queries & advantage computation for RLOO
            with torch.no_grad(), unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                queries = tokenizer(data["queries"], padding=True, return_tensors="pt")["input_ids"].to(device)
                final_responses = tokenizer(data["responses"], padding=True, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                queries = queries.repeat_interleave(args.rloo_k, 0)
                final_responses = final_responses.repeat_interleave(args.rloo_k, 0)
                context_length = queries.shape[1]
                accelerator.print("====RL sampling stage====")
                query_responses, responses, truncated_responses, response_ending_indices = [], [], [], []
                query_ppresponses, query_ppresponse_finals, query_context_lengths, reward_context_lengths = [], [], [], []
                logprobs, ref_logprobs, rewards = [], [], []
                for i in range(0, queries.shape[0], args.rollout_batch_size):
                    queries_rb = queries[i : i + args.rollout_batch_size]
                    final_responses_rb = final_responses[i : i + args.rollout_batch_size]
                    query_response_rb, _ = generate(
                        unwrapped_model,
                        queries_rb,
                        tokenizer.pad_token_id,
                        generation_config,
                    )

                    del _
                    gc.collect()
                    torch.cuda.empty_cache()

                    response_rb = query_response_rb[:, context_length:]

                    # 1.2 Reward & Advantage computation
                    # 1. Truncate response after the first occurrence of `stop_token_id`
                    # 2. Concatenation and Re-arrangement of query + truncated intermediate response + final_response
                    # 3. Run reward model on the query + postprocessed intermediate response + final_response
                    # 4. Filter response. Ensure that the sample contains stop_token_id
                    # NOTE: Without processing, concat (query + postprocessed intermediate response + final_response) should be something like:
                    #                                 <pad><bos>{query}, {reasoning}<pad>, <pad><end_of_thought>{final_answer}<eos>
                    # where <pad> is 0 to multiple pad tokens.
                    # But model.forward() would also not working with pad_tokens in the middle
                    # A better solution: rearrange all pad tokens to the beginning of each row while keeping the rest untouched, hence giving a batch like
                    #                                     <pad>, <bos>{query}, {reasoning}, <end_of_thought>{final_answer}<eos>
                    # since final_answer will always stay at the end of the sequence, first_true_indices(final_answer != pad_token_id) gives the pos of actual final_answer
                    # Hence ``first_true_indices(final_answer != pad_token_id) - 1 + query_ppresponse.shape[1]`` is the context length

                    truncated_responses_rb, response_ending_indices_rb = (
                        truncate_response(
                            args.stop_token_ids, tokenizer.pad_token_id, response_rb
                        )
                    )
                    query_ppresponses_rb = pad_mask_rearrange(
                        torch.cat((queries_rb, truncated_responses_rb), 1),
                        tokenizer.pad_token_id
                    )
                    query_context_lengths_rb = (query_ppresponses_rb.shape[1] - (response_ending_indices_rb + 1)).unsqueeze(1)

                    # penalty computation
                    logprobs_rb = get_logprob_reward(
                        model,
                        tokenizer.pad_token_id,
                        query_ppresponses_rb,
                        query_context_lengths_rb,
                    )
                    ref_logprobs_rb = get_logprob_reward(
                        ref_policy,
                        tokenizer.pad_token_id,
                        query_ppresponses_rb,
                        query_context_lengths_rb,
                    )

                    query_ppresponse_finals_rb = pad_mask_rearrange(
                        torch.cat((query_ppresponses_rb, final_responses_rb), 1),
                        tokenizer.pad_token_id
                    )
                    reward_context_lengths_rb = (
                        query_ppresponses_rb.shape[1]
                        + first_true_indices(final_responses_rb != tokenizer.pad_token_id)
                        - 1
                    ).unsqueeze(1)

                    rewards_rb = get_logprob_reward(
                        model,
                        tokenizer.pad_token_id,
                        query_ppresponse_finals_rb,
                        reward_context_lengths_rb,
                    )

                    query_responses.append(query_response_rb)
                    responses.append(response_rb)
                    truncated_responses.append(truncated_responses_rb)
                    response_ending_indices.append(response_ending_indices_rb)
                    query_ppresponses.append(query_ppresponses_rb)
                    query_ppresponse_finals.append(query_ppresponse_finals_rb)
                    query_context_lengths.append(query_context_lengths_rb)
                    reward_context_lengths.append(reward_context_lengths_rb)
                    logprobs.append(logprobs_rb)
                    ref_logprobs.append(ref_logprobs_rb)
                    rewards.append(rewards_rb)

                query_responses = torch.cat(query_responses, 0)
                responses = torch.cat(responses, 0)
                truncated_responses = torch.cat(truncated_responses, 0)
                response_ending_indices = torch.cat(response_ending_indices, 0)
                query_ppresponses = torch.cat(query_ppresponses, 0)
                query_ppresponse_finals = torch.cat(query_ppresponse_finals, 0)
                query_context_lengths = torch.cat(query_context_lengths, 0)
                reward_context_lengths = torch.cat(reward_context_lengths, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                rewards = torch.cat(rewards, 0)

                # NOTE: another choice is to add penalty over the trajectories of the same prompt
                contain_stop_token = torch.any(truncated_responses == tokenizer.pad_token_id, dim=-1)
                if args.non_stop_penalty:
                    rewards = torch.where(contain_stop_token, rewards, args.penalty_reward_value * rewards.mean())
                    # mean_rewards = rewards.reshape(-1, args.rloo_k).mean(dim=1)
                    # rewards = torch.where(
                    #     contain_stop_token,
                    #     rewards,
                    #     args.penalty_reward_value * mean_rewards.repeat_interleave(args.rloo_k)
                    # )

                # Advantage computation
                kl_penalty = -args.kl_coef * (logprobs - ref_logprobs)
                mean_kl_penalty = kl_penalty.mean().detach()
                adjusted_rewards = rewards + kl_penalty

                # vectorized RLOO advantages implementation
                adjusted_rewards = adjusted_rewards.reshape(-1, args.rloo_k) # NOTE: shape (N, K), every row is rewards for one prompt
                baseline = (adjusted_rewards.sum(1, keepdim=True) - adjusted_rewards) / (args.rloo_k - 1)

                advantages = adjusted_rewards - baseline
                advantages = advantages.flatten()

                if args.num_evaluations > 0 and (update - 1) % self.evaluation_freq == 0:
                    self.log_training_samples(
                        queries,
                        responses,
                        truncated_responses,
                        query_ppresponse_finals,
                        final_responses,
                        rewards,
                        advantages,
                    )

                del logprobs, ref_logprobs, rewards, baseline, kl_penalty
                gc.collect()
                torch.cuda.empty_cache()

            # Part 2: Training loops
            accelerator.print("====Training stage====")
            local_batch_indices = np.random.permutation(args.local_batch_size)
            gradient_accumulation_idx = 0
            for micro_batch_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
                with accelerator.accumulate(model):
                    micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                    micro_batch_inds = local_batch_indices[micro_batch_start:micro_batch_end]
                    advantage_mb = advantages[micro_batch_inds]
                    query_ppresponses_mb = query_ppresponses[micro_batch_inds]
                    query_ppresponse_finals_mb = query_ppresponse_finals[micro_batch_inds]
                    query_context_length_mb = query_context_lengths[micro_batch_inds]
                    reward_context_length_mb = reward_context_lengths[micro_batch_inds]

                    # 2.1  Update policy model q_phi with RLOO
                    # 2.1.1  Compute RLOO Loss
                    rationale_logprobs = get_logprob_reward(
                        model,
                        tokenizer.pad_token_id,
                        query_ppresponses_mb,
                        query_context_length_mb,
                    )
                    rloo_losses = -advantage_mb * rationale_logprobs
                    rloo_loss = rloo_losses.mean()

                    # 2.1.2  Compute metrics for RLOO
                    with torch.no_grad():
                        rloo_loss_stats[gradient_accumulation_idx] = rloo_loss

                    # 2.2  Update reward model p_theta with SFT
                    # reward_sft_loss = - log_softmax(reward_model(final_response | query+response)

                    # 2.2.1  Compute SFT loss
                    rewards = get_logprob_reward(
                        model,
                        tokenizer.pad_token_id,
                        query_ppresponse_finals_mb,
                        reward_context_length_mb,
                    )
                    sft_loss = -rewards.mean()

                    # 2.2.2  Compute metrics for SFT
                    with torch.no_grad():
                        sft_loss_stats[gradient_accumulation_idx] = sft_loss

                    # 2.3  Gradient descent
                    loss = rloo_loss + args.sft_penalty * sft_loss
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                gradient_accumulation_idx += 1
                # 2.4  del everything and empty cache
                # fmt: off
                del (
                    rationale_logprobs, rloo_losses, rloo_loss, loss, advantage_mb, query_ppresponses_mb,
                    query_ppresponse_finals_mb, reward_context_length_mb, sft_loss
                )
                # fmt: on
                gc.collect()
                torch.cuda.empty_cache()

            # Part 3: Logging metrics
            with torch.no_grad():
                response_lengths = (response_ending_indices + 1).float().mean()
                num_finished_thoughts = contain_stop_token.sum(0)
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl_penalty"] = self.accelerator.gather(mean_kl_penalty).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(adjusted_rewards).mean().item()
                metrics["objective/advantage_avg"] = self.accelerator.gather(advantages).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(rloo_loss_stats).mean().item()
                metrics["loss/sft_avg"] = self.accelerator.gather(sft_loss_stats).mean().item()
                metrics["val/num_stop_tokens"] = self.accelerator.gather(num_finished_thoughts).sum().item()
                metrics["val/perc_finished_thoughts"] = metrics["val/num_stop_tokens"]/(args.local_batch_size * self.accelerator.num_processes)
                metrics["val/avg_rationale_length"] = self.accelerator.gather(response_lengths).mean().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

                del (
                    mean_kl_penalty, adjusted_rewards, num_finished_thoughts, response_lengths, response_ending_indices,
                    query_ppresponse_finals, contain_stop_token, reward_context_lengths, query_responses,
                    responses, advantages
                )
                gc.collect()
                torch.cuda.empty_cache()

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            if args.num_evaluations > 0 and (update - 1) % self.evaluation_freq == 0:
                accelerator.print("====Evaluation stage====")
                self.evaluation()
            gc.collect()
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
