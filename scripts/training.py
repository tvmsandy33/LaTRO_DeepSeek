from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from src.trainer import (
    LatroTrainer,
    TrainerConfig,
)
from src.utils.data_utils import prepare_dataset_gsm8k, prepare_dataset_arc
from src.utils.trainer_utils import get_end_token, set_pad_token


parser = HfArgumentParser(TrainerConfig)
config: TrainerConfig = parser.parse_args_into_dataclasses()[0]

if PartialState().is_local_main_process:
    # wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="implicit-reasoning",
        name=config.run_name,
    )


################
# Model & Tokenizer
################
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name_or_path,
    padding_side="left",
)

policy = AutoModelForCausalLM.from_pretrained(
    config.model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)
ref_policy = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

# Manipulate the tokens
if tokenizer.pad_token is None:
    set_pad_token(tokenizer)

if config.stop_token is not None and config.stop_token_ids is None:
    match config.stop_token:
        case "eos":
            config.stop_token_ids = [tokenizer.eos_token_id]
        case "pad":
            config.stop_token_ids = [tokenizer.pad_token_id]
        case "both":
            config.stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]

if config.stop_seqs is not None:
    no_prefix_tokenizer =  AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        padding_side="left",
        add_prefix_space=False
    )
    stop_seqs_ids = no_prefix_tokenizer(config.stop_seqs, add_special_tokens=False)["input_ids"]
    config.stop_token_ids.extend(stop_seqs_ids)

if (end_token := get_end_token(tokenizer)) not in config.stop_token_ids:
    config.stop_token_ids.append(tokenizer.convert_tokens_to_ids(end_token))

################
# Dataset
################
match config.dataset_name:
    case "gsm8k":
        raw_datasets = load_dataset(path="openai/gsm8k", name="main")
        prepare_dataset = prepare_dataset_gsm8k
    case "arc_challenge":
        raw_datasets = load_dataset(path="allenai/ai2_arc", name="ARC-Challenge")
        prepare_dataset = prepare_dataset_arc
    case _:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")
if config.sanity_check:
    for key in raw_datasets:
        raw_datasets[key] = raw_datasets[key].select(range(1000))

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

# Compute that only on the main process for faster data processing.
# see: https://github.com/huggingface/trl/pull/1255
with PartialState().local_main_process_first():
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

################
# Training
################
trainer = LatroTrainer(
    config=config,
    tokenizer=tokenizer,
    policy=policy,
    ref_policy=ref_policy,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train(resume_from_checkpoint=config.checkpoint_path)
torch.cuda.empty_cache()
################
# Evaluation
################
trainer.evaluation()
trainer.save_model(config.output_dir)
if config.push_to_hub:
    trainer.push_to_hub()
