TIME=$(date '+%Y-%m-%d_%H-%M-%S')
RUN_NAME=llama3.1_8b_gsm8k_$TIME

RESPONSE_LENGTH=500
NUM_EPOCHS=6

accelerate launch --config_file configs/8gpu.yaml \
    scripts/training.py \
    --output_dir "model/$RUN_NAME/" \
    --run_name $RUN_NAME \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset_name "gsm8k" \
    --sanity_check false \
    --num_train_epochs $NUM_EPOCHS \
    --num_evaluations $NUM_EPOCHS \
    --gradient_checkpointing false \
    --per_device_eval_batch_size 64 \
    --rollout_batch_size 48 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 12 \
    --learning_rate 5e-7 \
    --rloo_k 16 \
    --kl_coef 0.05 \
    --response_length $RESPONSE_LENGTH 

# TIME=$(date '+%Y-%m-%d_%H-%M-%S')
# RUN_NAME=mistral_7b_gsm8k_$TIME

# RESPONSE_LENGTH=500
# NUM_EPOCHS=6

# accelerate launch --config_file configs/8gpu.yaml \
#     scripts/training.py \
#     --output_dir "model/$RUN_NAME/" \
#     --run_name $RUN_NAME \
#     --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
#     --dataset_name "gsm8k" \
#     --sanity_check false \
#     --num_train_epochs $NUM_EPOCHS \
#     --num_evaluations $NUM_EPOCHS \
#     --gradient_checkpointing false \
#     --per_device_eval_batch_size 48 \
#     --rollout_batch_size 48 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 12 \
#     --learning_rate 5e-7 \
#     --rloo_k 16 \
#     --kl_coef 0.05 \
#     --response_length $RESPONSE_LENGTH \

# TIME=$(date '+%Y-%m-%d_%H-%M-%S')
# RUN_NAME=phi3.5_mini_gsm8k_$TIME

# RESPONSE_LENGTH=500
# NUM_EPOCHS=6

# accelerate launch --config_file configs/8gpu.yaml \
#     scripts/training.py \
#     --output_dir "model/$RUN_NAME/" \
#     --run_name $RUN_NAME \
#     --model_name_or_path microsoft/Phi-3.5-mini-instruct \
#     --dataset_name "gsm8k" \
#     --sanity_check false \
#     --num_train_epochs $NUM_EPOCHS \
#     --num_evaluations $NUM_EPOCHS \
#     --gradient_checkpointing false \
#     --per_device_eval_batch_size 48 \
#     --rollout_batch_size 48 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 12 \
#     --learning_rate 5e-7 \
#     --rloo_k 16 \
#     --kl_coef 0.05 \
#     --response_length $RESPONSE_LENGTH \

