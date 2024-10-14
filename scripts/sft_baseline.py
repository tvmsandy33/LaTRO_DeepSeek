import os
from dotenv import find_dotenv, load_dotenv
from typing import Optional, Literal
import torch
import wandb
from accelerate import PartialState
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from src.utils.trainer_utils import set_pad_token

load_dotenv(find_dotenv())


@dataclass
class ScriptArgs(TrainingArguments):
    # common config
    output_dir: Optional[str] = None
    """output directory"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    sanity_check: bool = False
    """wether to run in debug mode"""
    dataset_name: Optional[Literal["gsm8k", "arc_challenge"]] = "gsm8k"
    """Name of the dataset"""
    # SFT config
    model_name_or_path: str = "EleutherAI/pythia-160m"
    """the path to the policy model"""
    checkpoint_path: Optional[str] = None
    """path to the checkpoint"""
    num_train_epochs: int = 1
    """number of epochs to train"""
    per_device_train_batch_size: Optional[int] = 16
    """batch size per gpu used during training"""
    per_device_eval_batch_size: Optional[int] = 16
    """batch size per gpu used during evaluation"""
    gradient_accumulation_steps: Optional[int] = 1
    """Number of updates steps to accumulate the gradients for, before performing a backward/update pass."""
    learning_rate: float = 5e-7
    """Initial learning rate"""


parser = HfArgumentParser(ScriptArgs)
config: ScriptArgs = parser.parse_args_into_dataclasses()[0]

args = SFTConfig(
    output_dir=config.output_dir,
    run_name=config.run_name,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    bf16=True,
    max_grad_norm=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=42,
)

if PartialState().is_local_main_process:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="implicit-reasoning",
        name=config.run_name,
        tags=["sft_baseline"],
    )

################
# Model & Tokenizer
################
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name_or_path,
    padding_side="left",
)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name_or_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

data_collator = None

if tokenizer.pad_token is None:
    set_pad_token(tokenizer)

match tokenizer.name_or_path:
    case "microsoft/Phi-3.5-mini-instruct":
        tokenizer.eos_token = "<|end|>"
        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template="<|user|>",
            response_template="<|assistant|>",
            tokenizer=tokenizer,
        )
    case "meta-llama/Meta-Llama-3.1-8B-Instruct":
        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template="<|start_header_id|>user<|end_header_id|>",
            response_template="<|start_header_id|>assistant<|end_header_id|>",
            tokenizer=tokenizer,
        )
    case "mistralai/Mistral-7B-Instruct-v0.3":
        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template="[INST]",
            response_template="[/INST]",
            tokenizer=tokenizer,
        )


def prepare_dataset_gsm8k(dataset: Dataset):
    """
    preprocess gsm8k dataset
    """
    messages = [
        [
            {"role": "user", "content": x},
            {"role": "assistant", "content": y.replace("####", "The answer is")},
        ]
        for x, y in zip(dataset["question"], dataset["answer"])
    ]
    dataset = Dataset.from_dict({"messages": messages})
    return dataset

def prepare_dataset_arc(dataset: Dataset):
    """
    preprocess arc datasets
    """
    groundtruth = [
        text
        for choices, answer_key in zip(dataset["choices"], dataset["answerKey"])
        for text, label in zip(choices["text"], choices["label"])
        if label == answer_key
    ]
    responses = [f"The answer is {x}." for x in groundtruth]
    messages = [
        [
            {"role": "user", "content": f"""{question}\nOptions: {choices["text"]}\nChoose one from available options and return your answer like `The answer is {{final_answer}}`."""},
            {"role": "assistant", "content": response}
        ]
        for question, choices, response in zip(dataset["question"], dataset["choices"], responses)
    ]

    dataset = Dataset.from_dict({"messages": messages})
    return dataset

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
    train_dataset = prepare_dataset(train_dataset)
    eval_dataset = prepare_dataset(eval_dataset)
if PartialState().is_local_main_process:
    print(train_dataset[0])
################
# Training
################
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args,
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=config.checkpoint_path)
torch.cuda.empty_cache()
################
# Evaluation
################
trainer.save_model(args.output_dir)
