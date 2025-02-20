import json
from datasets import Dataset
from transformers import AutoTokenizer
from .trainer_utils import get_end_token

FEW_SHOT_EXAMPLES_GSM8K = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.
"""


def prepare_dataset_gsm8k(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    use_few_shot: bool = False,
):
    """
    preprocess gsm8k dataset
    """
    end_token = get_end_token(tokenizer)
    responses = dataset["answer"]
    if tokenizer.chat_template is None:
        if use_few_shot:
            queries = [
                f"""{FEW_SHOT_EXAMPLES_GSM8K}\n{x}\nLet's think step by step."""
                for x in dataset["question"]
            ]
        else:
            queries = [
                f"""{x}\nLet's think step by step.""" for x in dataset["question"]
            ]
    else:
        if use_few_shot:
            queries = [
                [
                    {
                        "role": "user",
                        "content": FEW_SHOT_EXAMPLES_GSM8K
                        + "\n"
                        + x
                        + " Let's think step by step.",
                    }
                ]
                for x in dataset["question"]
            ]
        else:
            queries = [
                [{"role": "user", "content": x + " Let's think step by step."}]
                for x in dataset["question"]
            ]
        queries = tokenizer.apply_chat_template(
            queries, tokenize=False, add_generation_prompt=True
        )
    rationale_responses = [x.split("#### ") for x in responses]
    rationales = [x[0] for x in rationale_responses]
    responses = [f"The answer is {x[1]}." + end_token for x in rationale_responses]
    groundtruth = [x[1] for x in rationale_responses]
    dataset = Dataset.from_dict(
        {"queries": queries, "responses": responses, "groundtruth": groundtruth, "rationales": rationales}
    )
    return dataset


FORMAT_ARC_SFT = """
The output MUST strictly follow the structure and format described below:

1. The answer should be strictly one of the options given in the question
2. Do not include any text after you choose an option.
**Example:**
Question: (question here)
Options: [option1, option2, option3]
The answer is: (one of the options).
"""


def prepare_dataset_arc(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    sft: bool = False,
):
    """
    preprocess ARC-Challenge dataset
    """
    end_token = get_end_token(tokenizer)
    if tokenizer.chat_template is None:
        if sft:
            queries = [
                f"""Question: {question}\nOptions: {choices["text"]}\nAnswer:"""
                for question, choices in zip(dataset["question"], dataset["choices"])
            ]
        else:
            queries = [
                f"""Question: {question}\nOptions: {choices["text"]}\nAnswer: Let's think step by step."""
                for question, choices in zip(dataset["question"], dataset["choices"])
            ]
    else:
        if sft:
            queries = [
                [
                    {
                        "role": "user",
                        "content": f"""{FORMAT_ARC_SFT}\nQuestion: {question}\nOptions: {choices["text"]}\n""",
                    }
                ]
                for question, choices in zip(dataset["question"], dataset["choices"])
            ]
        else:
            queries = [
                [
                    {
                        "role": "user",
                        "content": f"""Question: {question}\nOptions: {choices["text"]}\nLet's think step by step first then print your answer like 'The answer is (one of the options)'.""",
                    }
                ]
                for question, choices in zip(dataset["question"], dataset["choices"])
            ]
        queries = tokenizer.apply_chat_template(
            queries, tokenize=False, add_generation_prompt=True
        )
    groundtruth = [
        text
        for choices, answer_key in zip(dataset["choices"], dataset["answerKey"])
        for text, label in zip(choices["text"], choices["label"])
        if label == answer_key
    ]
    responses = [f"The answer is {x}." + end_token for x in groundtruth]
    dataset = Dataset.from_dict(
        {"queries": queries, "responses": responses, "groundtruth": groundtruth}
    )
    return dataset
