import gc
import json
from dataclasses import dataclass, asdict
from typing import Optional, Literal
from collections import defaultdict
from datetime import datetime

import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    GenerationConfig,
    set_seed,
)
from datasets import load_dataset
from trl.models.utils import unwrap_model_for_generation
from peft import PeftModel

from src.utils.data_utils import (
    prepare_dataset_gsm8k,
    prepare_dataset_arc,
)
from src.utils.eval_utils import (
    eval_func_gsm8k,
    eval_func_arc,
)
from src.utils.trainer_utils import (
    get_end_token,
    set_pad_token,
    batch_generation,
    truncate_response,
)


@dataclass
class EvaluationArgs:
    base_model_name_or_path: str
    """name/path of the baseline model"""
    eval_model_name_or_path: str
    """name/path of the model checkpoint to evaluate"""

    dataset_name: Optional[
        Literal["gsm8k_0shot", "gsm8k_8shot", "arc_challenge"]
    ] = "gsm8k_0shot"
    """name of the eval dataset"""
    eval_batch_size: int = 16
    """batch size of the eval dataloader"""

    response_length: int = 200
    """maximum response length of the generation"""
    temperature: float = 0.0
    """temperature used for generation"""
    seed: int = 42
    """seed used for eval"""

    stop_token: Optional[Literal["eos", "pad", "both"]] = "both"
    """the stop token"""
    stop_token_ids: list[int] | None = None
    """the stop token id or stop token ids, will overwrite stop_token"""


def main():
    parser = HfArgumentParser(EvaluationArgs)
    args: EvaluationArgs = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)

    # Initialize the Accelerator
    accelerator = Accelerator()
    # Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, padding_side="left"
    )
    if tokenizer.pad_token is None:
        set_pad_token(tokenizer)
    if args.stop_token is not None and args.stop_token_ids is None:
        match args.stop_token:
            case "eos":
                args.stop_token_ids = [tokenizer.eos_token_id]
            case "pad":
                args.stop_token_ids = [tokenizer.pad_token_id]
            case "both":
                args.stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    if (end_token := get_end_token(tokenizer)) not in args.stop_token_ids:
        args.stop_token_ids.append(tokenizer.convert_tokens_to_ids(end_token))

    model = AutoModelForCausalLM.from_pretrained(
        args.eval_model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()

    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        temperature=(args.temperature + 1e-7),
        eos_token_id=args.stop_token_ids,
        pad_token_id=tokenizer.pad_token_id,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    # Eval Dataloader
    match args.dataset_name:
        case "gsm8k_0shot" | "gsm8k_8shot":
            raw_datasets = load_dataset(path="openai/GSM8K", name="main")
            eval_dataset = prepare_dataset_gsm8k(
                raw_datasets["test"],
                tokenizer,
                use_few_shot=args.dataset_name == "gsm8k_8shot",
            )
            eval_func = eval_func_gsm8k
        case "arc_challenge":
            raw_datasets = load_dataset(path="allenai/ai2_arc", name="ARC-Challenge")
            eval_dataset = prepare_dataset_arc(
                raw_datasets["test"],
                tokenizer,
                sft="sft" in args.eval_model_name_or_path,
            )
            eval_func = eval_func_arc
        case _:
            raise ValueError(f"Dataset {args.dataset_name} not supported")
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False
    )

    # Prepare model for distributed environment
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()
    device = accelerator.device
    table = defaultdict(list)

    for batch in tqdm(eval_dataloader):
        queries = tokenizer(batch["queries"], padding=True, return_tensors="pt")["input_ids"].to(device)
        groundtruth = tokenizer(batch["groundtruth"], padding=True, return_tensors="pt")["input_ids"].to(device)
        context_length = queries.shape[1]
        # generation
        with (
            unwrap_model_for_generation(model, accelerator) as unwrapped_model,
            torch.no_grad(),
        ):
            eval_generation, _ = batch_generation(
                unwrapped_model,
                queries,
                queries.shape[0],
                tokenizer.pad_token_id,
                generation_config,
            )
        del _
        torch.cuda.empty_cache()
        # filter responses and truncate stop tokens
        eval_response = eval_generation[:, context_length:]
        eval_response, _ = truncate_response(
            args.stop_token_ids, tokenizer.pad_token_id, eval_response
        )

        # gather data
        table["query"].extend(
            gather_object(tokenizer.batch_decode(queries, skip_special_tokens=True))
        )
        table["model_response"].extend(
            gather_object(
                tokenizer.batch_decode(eval_response, skip_special_tokens=True)
            )
        )
        table["groundtruth"].extend(
            gather_object(tokenizer.batch_decode(groundtruth, skip_special_tokens=True))
        )
        del queries, groundtruth, eval_generation, eval_response, _
        torch.cuda.empty_cache()
        gc.collect()

    # create dataframe
    df_result = pd.DataFrame(table)
    df_result = df_result[:len(eval_dataset)]
    df_result["result"] = df_result.apply(eval_func, axis=1)

    count = int(df_result["result"].sum())
    acc = count / len(df_result)
    accelerator.print(f"Correctly answered questions: {count}; Accuracy: {acc:.2%}")

    if PartialState().is_main_process:
        base_model_name = args.base_model_name_or_path.split("/")[-1]
        checkpoint_name = (
            args.eval_model_name_or_path.replace("/", "_")
            if args.eval_model_name_or_path != args.base_model_name_or_path
            else "baseline"
        )
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_dict = {
            "result": df_result.to_dict(orient="records"),
            "correct_count": count,
            "acc": acc,
        }
        result_dict.update(asdict(args))
        with open(
            f"./evaluation/{args.dataset_name}_{base_model_name}:{checkpoint_name}_{time_stamp}.json",
            "w",
        ) as out_file:
            json.dump(result_dict, out_file, indent=4)


if __name__ == "__main__":
    main()
