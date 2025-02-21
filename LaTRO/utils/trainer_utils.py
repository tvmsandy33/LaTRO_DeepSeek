from dataclasses import dataclass
import torch
import gc

from trl.trainer.utils import first_true_indices
from transformers import AutoTokenizer
from transformers.trainer import TrainerState

@dataclass
class OnlineTrainerState(TrainerState):
    episode: int = 0


def get_end_token(tokenizer: AutoTokenizer) -> str:
    """
    Get the real end token of an instructed model.
    """
    match tokenizer.name_or_path:
        case "microsoft/Phi-3.5-mini-instruct":
            return "<|end|>"
        case _:
            return tokenizer.eos_token


def set_pad_token(tokenizer: AutoTokenizer):
    """
    Set the pad token other than EOS of a tokenizer if it is not set.
    """
    match tokenizer.name_or_path:
        # NOTE: DO NOT set tokenizer.pad_token = tokenizer.eos_token if the tokenizer doesn't have a pad token
        case "meta-llama/Meta-Llama-3-8B-Instruct" | "meta-llama/Meta-Llama-3.1-8B-Instruct":
            tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
        case "mistralai/Mistral-7B-Instruct-v0.3":
            tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
        case _:
            raise ValueError(
                f"Unsupported tokenizer {tokenizer.name_or_path}, please check the tokenizer's vocabulary and add properly the pad token"
            )


def forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
) -> torch.nn.Module:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `torch.nn.Module`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def generate(
    lm_backbone: torch.nn.Module,
    queries: torch.Tensor,
    pad_token_id: int,
    generation_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`dict`):
            The configuration dictionary for generation settings.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: dict,
):
    query_responses = []
    logitss = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)
    return torch.cat(query_responses, 0), torch.cat(logitss, 0)


def detect_stop_sequence_starts(
    input_ids: torch.Tensor, stop_seq_ids: list[int]
) -> torch.Tensor:
    """
    Detects if each element in input_ids is the start of the stop_seq_ids sequence.

    Args:
        input_ids (torch.Tensor): Tensor of shape [batch_size, seq_len].
        stop_seq_ids (List[int]): List of token ids representing the stop sequence.

    Returns:
        torch.Tensor: A boolean tensor of shape [batch_size, seq_len] where each element is True
                      if the token is the start of the stop_seq_ids sequence.
    """
    # Convert stop_seq_ids to a tensor
    stop_seq_tensor = torch.tensor(stop_seq_ids, device=input_ids.device)
    stop_seq_len = len(stop_seq_ids)

    # Get batch size and sequence length
    batch_size, seq_len = input_ids.shape

    # Initialize a tensor to hold detection results
    detected = torch.zeros(
        (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
    )

    # Slide a window over each sequence in the batch
    for i in range(seq_len - stop_seq_len + 1):
        # Extract a window of size len(stop_seq_ids) from input_ids
        window = input_ids[:, i : i + stop_seq_len]

        # Check if the window matches the stop sequence
        match = torch.all(window == stop_seq_tensor, dim=1)

        # Update the first position of the match to True in the detected tensor
        detected[:, i] = match

    return detected


def truncate_response(
    stop_token_ids: list[int | list[int]],
    pad_token_id: int,
    responses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates the responses at the first occurrence of the stop token,
    filling the stop token and the rest with pad tokens.

    Args:
        stop_token_ids (`list[int]`):
            The token IDs representing the stop token where truncation occurs.
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses.
        responses (`torch.Tensor`):
            The tensor containing the responses to be truncated.
            Assuming after the first occurence of stop_token, the responses will be useless

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`:
            (The truncated responses tensor with pad tokens filled starting the stop token, ending index of the actual response)
    """
    masks = []
    for stop_token_or_seq in stop_token_ids:
        if isinstance(stop_token_or_seq, int):
            masks.append(responses == stop_token_or_seq)
        elif isinstance(stop_token_or_seq, list):
            masks.append(detect_stop_sequence_starts(responses, stop_token_or_seq))
        else:
            raise ValueError(f"stop_token_ids can only contain int or list[int], found {type(stop_token_or_seq)}")
    mask = torch.stack(masks).sum(0, dtype=bool)
    trunc_idxs = first_true_indices(mask)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    responses = torch.masked_fill(
        responses, idxs >= trunc_idxs.unsqueeze(-1), pad_token_id
    )
    return responses, trunc_idxs - 1


def pad_mask_rearrange(input_ids: torch.Tensor, pad_token_id: int):
    """rearrange each row in the input_ids so that all pad tokens are in the beginning and the order of rest tokens is kept."""
    mask = (input_ids == pad_token_id).long()
    rearrange_index = torch.argsort(mask, dim=1, descending=True, stable=True)
    rearranged = torch.gather(input_ids, 1, rearrange_index)
    return rearranged


def get_logprob_reward(
    model: torch.nn.Module,
    pad_token_id: int,
    input_ids: torch.Tensor,
    context_lengths: int | torch.Tensor,
) -> torch.Tensor:
    """Give the logprobs of model(response|context) as a reward function with each sequence in the batch left padded with different context lengths"""
    if isinstance(context_lengths, int):
        context_lengths = torch.full((input_ids.shape[0], 1), context_lengths, dtype=torch.long, device=input_ids.device)
    logits = forward(model, input_ids, pad_token_id).logits  # Raw logits [batch_size, seq_len, vocab_size]

    torch.cuda.empty_cache()
    gc.collect()

    logprobs = torch.log_softmax(logits, dim=-1)  # transform to log probs

    del logits
    torch.cuda.empty_cache()
    gc.collect()

    sequence_logprob = torch.gather(logprobs[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # select next token probabilities of each token in the reward_all_logprob
    del logprobs
    torch.cuda.empty_cache()
    gc.collect()

    # NOTE: Now we only need to sum the part from actual final_response using a mask from reward_context_length
    indices = torch.arange(sequence_logprob.shape[1], device=sequence_logprob.device).repeat(sequence_logprob.shape[0], 1)
    mask = indices >= context_lengths - 1 # starting at the end of the context
    reward = (sequence_logprob * mask).sum(1)

    del sequence_logprob, indices, mask
    torch.cuda.empty_cache()
    gc.collect()
    return reward
