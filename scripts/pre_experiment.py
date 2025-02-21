import os
import json
from dotenv import find_dotenv, load_dotenv
from collections import defaultdict

load_dotenv(find_dotenv())

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from LaTRO.utils.data_utils import prepare_dataset_gsm8k
from LaTRO.utils.trainer_utils import set_pad_token, get_logprob_reward, pad_mask_rearrange, first_true_indices

model_names = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]


def main(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    if tokenizer.pad_token is None:
        set_pad_token(tokenizer)

    raw_datasets = load_dataset(path="openai/gsm8k", name="main")
    train_dataset = prepare_dataset_gsm8k(raw_datasets["train"], tokenizer)

    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(42)
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    results = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            queries = tokenizer(batch["queries"], padding=True, return_tensors="pt")["input_ids"].to(device)
            query_rationales = tokenizer(
                [x + y for x, y in zip(batch["queries"], batch["rationales"])],
                padding=True,
                return_tensors="pt",
            )["input_ids"].to(device)
            responses = tokenizer(
                batch["responses"],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(device)

            input_no_rationale = pad_mask_rearrange(torch.cat((queries, responses), 1), tokenizer.pad_token_id)
            context_lengths_no_rationale = (queries.shape[1] + (first_true_indices(responses != tokenizer.pad_token_id) - 1)).unsqueeze(1)
            logprobs_no_rationale = get_logprob_reward(model, tokenizer.pad_token_id, input_no_rationale, context_lengths_no_rationale)

            input_with_rationale = pad_mask_rearrange(
                torch.cat((query_rationales, responses), 1), tokenizer.pad_token_id
            )
            context_lengths_with_rationale = (
                query_rationales.shape[1] + (first_true_indices(responses != tokenizer.pad_token_id) - 1)
            ).unsqueeze(1)
            logprobs_with_rationale = get_logprob_reward(
                model,
                tokenizer.pad_token_id,
                input_with_rationale,
                context_lengths_with_rationale,
            )

            results["logprobs_no_rationale"].extend(accelerator.gather(logprobs_no_rationale).cpu().numpy().tolist())
            results["logprobs_with_rationale"].extend(accelerator.gather(logprobs_with_rationale).cpu().numpy().tolist())
    results["logprobs_avg_no_rationale"] = np.mean(results["logprobs_no_rationale"])
    results["logprobs_avg_with_rationale"] = np.mean(results["logprobs_with_rationale"])
    with open(f"experiment/logprobs_diff_{model_name.replace('/', '_')}.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    for model_name in model_names:
        main(model_name)
