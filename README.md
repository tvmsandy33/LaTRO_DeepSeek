Codes for [Latent Reasoning Optimization](https://arxiv.org/abs/2411.04282) (LaTRO).

# Quickstart

## Installation
Under the repo path, run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
*Note*: If you can't install flash attention, remove it from the `requriements.txt` and follow instructions [here](https://github.com/Dao-AILab/flash-attention).

(Optional) Prepare an `.env` under the root folder
```
# accessing gated huggingface models
HF_TOKEN=""
# for logging in wandb
WANDB_API_KEY=""
WANDB_PROJECT=""
```

To use wandb, first run
```bash
wandb login --relogin --host=https://your_wandb_host
```
with your wandb api key.

## Run training and evaluation
Prepare your args in the corresponding `.sh` scripts first then run, see the scripts for details.
```bash
bash scripts/training.sh # training
bash scripts/evaluation.sh # eval
```

## How to add new datasets
1. Prepare a dataset preprocessing function, see `prepare_data_gsm8k` from `data_utils.py` as an example.
    1. The dataset (in transformers `Dataset` class) should contain 3 columns: `queries`, `responses` and `groundtruth`.
    2. `queries` contain the inputs plus a CoT trigger, like "Let's think step by step". And wrapped with the chat template of the model.
    3. `responses` contain the desired responses in an answer template, e.g. "The answer is XXX".
    4. `groundtruth` is the groundtruth for evaluation, e.g. answer to a math/multiple choice question, or a json containing function name and args.
    5. `end_of_thought_token: str = None` is an arg to be deprecated. Tentatively put it as a dummy arg.
2. Prepare an eval function in `eval_utils.py` (see examples therein)
3. Modify config definitions in `trainer_config.py` and `evaluation.py` to support your dataset
4. Include your data prep function in `training.py` and `evaluation.py`
5. Include your evaluation function in `evaluation.py` and `BaseTrainer.evaluation()` in `trainer.py`

## Important training args
See `implicit_reasoning/trainer/trainer_config.py` for detailes. The config class inherites the huggingface `TrainingArguments`.


Trainer meta args:
- `model_name_or_path`: name of the base model
- `checkpoint_path`: path to a checkpoint to resume, default to be `None` and default to train base model from scratch
- `gradient_checkpointing`: if to use gradient checkpointing, set to `false` unless you are reaaaally OOM.
- `dataset_name`: which dataset to use, default is `gsm8k`
- `sanity_check`: if true, will only run for a few data to debug
- `num_train_epochs`: epochs to train
- `num_evaluations`: set it > 0, and usually equal to num_train_epochs

Batch sizes:
- `per_device_eval_batch_size`: evaluation batch size
- `per_device_train_batch_size`: minibatch size in the training loop
- `gradient_accumulation_steps`: gradient accumulation steps
- `rollout_batch_size`: batch size during the mc sampling. default to be 16 to fit the MATH dataset. Can be larger if your dataset is smaller.

**IMPORTANT: make sure `per_device_train_batch_size * gradient_accumulation_steps` is a multiple of `rloo_k` (see below) for minibatch loops to run!**

The actual batch size from the dataloader is `per_device_train_batch_size * gradient_accumulation_steps / rloo_k`. Each of them will be sampled `rloo_k` times.
Then during the training loop, a for-loop of `gradient_accumulation_steps` will run on micro batches of `per_device_train_batch_size` samples.

MC Sampling args:
- `rloo_k`: number of Monte Carlo samples for each input datapoint in one global update.
- `response_length`: controls how many tokens to generate for rationale sampling (actual rationale will be shorter, thus we do truncation)

Rationale postprocessing args:
- `stop_token`: stop tokens used to truncate the rationale, can be `eos`, `pad` or `both`, default to use `both`.
- `stop_seqs`: a list of strings used to truncate the rationale, e.g. `"Answer: "` or `"The answer is"`

# Repo structure
```bash
├── README.md
├── configs # contains deepspeed and accelerate configs, use 8gpu.yaml by default
├── src
│   ├── trainer # contain different trainers
│   │   ├── __init__.py
│   │   ├── base_trainer.py # base trainer, use only for __init__()
│   │   ├── trainer_config.py # dataclass of training args
│   │   └── latro_trainer.py # the actual trainer
│   └── utils
│       ├── data_utils.py # utils for data processing
│       ├── eval_utils.py # utils for answer extraction and evaluation
│       └── trainer_utils.py # utils for training, tensor manipulation
├── scripts # training and evaluation python/bash scripts, tune your params in the .sh files
│   ├── evaluation.py # evaluation
│   ├── evaluation.sh
│   ├── sft_baseline.py # SFT training baslines
│   ├── sft_baseline.sh
│   ├── training.py # Training
│   └── training.sh
├── requirements.txt
└── setup.py
```

# Citation
```
@misc{chen2024languagemodelshiddenreasoners,
      title={Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding}, 
      author={Haolin Chen and Yihao Feng and Zuxin Liu and Weiran Yao and Akshara Prabhakar and Shelby Heinecke and Ricky Ho and Phil Mui and Silvio Savarese and Caiming Xiong and Huan Wang},
      year={2024},
      eprint={2411.04282},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2411.04282}, 
}
```

# Ethical Considerations
This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.
