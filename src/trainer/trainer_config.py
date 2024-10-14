import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import (
    TrainingArguments,
)


@dataclass
class TrainerConfig(TrainingArguments):
    """
    Config for reasoning trajectory optimization
    This class inherits from TrainingArguments, the args defined therein could be not listed here.
    """

    exp_name: str = field(
        default=os.path.basename(__file__)[: -len(".py")],
        metadata={"help": "The name of this experiment"},
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "A unique name of this run"}
    )
    sanity_check: bool = field(
        default=False, metadata={"help": "Whether to run in debug mode"}
    )

    model_name_or_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "The path to the policy model"},
    )
    checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "checkpoint path"}
    )

    total_episodes: Optional[int] = field(
        default=None,
        metadata={
            "help": "The total number of episodes in the dataset, will overwrite num_train_epochs"
        },
    )
    num_evaluations: int = field(
        default=10, metadata={"help": "The number of evaluations throughout training"}
    )
    per_device_eval_batch_size: int = field(
        default=20, metadata={"help": "Batch size per GPU used during evaluation"}
    )
    rollout_batch_size: int = field(
        default=16,
        metadata={
            "help": "Batch size during the mc sampling process, must be smaller than per_device_train_batch_size*gradient_accmulation_steps"
        },
    )

    response_length: int = field(
        default=200, metadata={"help": "The length of the response"}
    )
    stop_token: Optional[Literal["eos", "pad", "both"]] = field(
        default="both", metadata={"help": "The stop token"}
    )
    stop_token_ids: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": "The stop token id or stop token ids, will overwrite stop_token"
        },
    )
    stop_seqs: list[str] = field(
        default_factory=lambda: [
            "Answer:",
            " Answer:",
            "\nAnswer:",
            "The answer is",
            " The answer is",
            "\nThe answer is",
            "The answer is:",
            " The answer is:",
            "\nThe answer is:",
            " the answer is",
            "so the final answer is",
            "So, the answer is",
            "Therefore, the answer is",
            "The function call is:",
            "\nThe function call is:",
            "The function call is",
            "\nThe function call is",
            "the function call is",
        ],
        metadata={
            "help": "The list of stop sequences, like answer template for extraction"
        },
    )
    temperature: float = field(default=1, metadata={"help": "The sampling temperature"})
    learning_rate: float = field(
        default=5e-7, metadata={"help": "Initial learning rate of the optimizer"}
    )
    penalty_reward_value: int = field(
        default=2,
        metadata={
            "help": "The reward value penalty multiplier for responses that do not contain `stop_token_id`, will set the reward to `penalty_reward_value * reward.mean()`"
        },
    )
    non_stop_penalty: bool = field(
        default=True,
        metadata={
            "help": "Whether to penalize responses that do not contain `stop_token_id`"
        },
    )
    kl_coef: float = field(default=0.05, metadata={"help": "The KL coefficient"})
    rloo_k: int = field(
        default=16,
        metadata={
            "help": "REINFORCE Leave-One-Out (RLOO) number of online samples per prompt"
        },
    )
    sft_penalty: float = field(
        default=0, metadata={"help": "The level of SFT loss to add in the final loss"}
    )

    dataset_name: Optional[
        Literal[
            "gsm8k",
            "arc_challenge",
        ]
    ] = field(
        default="gsm8k",
        metadata={
            "help": "Path to the dataset, assuming it's already split into train/eval and contains columns ['queries', 'responses']"
        },
    )
    dataset_num_proc: int = field(
        default=4, metadata={"help": "Number of processes to preprocess the dataset"}
    )
