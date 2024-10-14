TIME=$(date '+%Y-%m-%d_%H-%M-%S')
RUN_NAME=mistral_7b_arc_challenge_sft_$TIME

accelerate launch --config_file configs/8gpu.yaml \
    scripts/sft_baseline.py \
    --output_dir "model/$RUN_NAME/" \
    --run_name $RUN_NAME \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset_name "arc_challenge" \
    --sanity_check false \
    --num_train_epochs 12 \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-7 \
    --use_lora false \
