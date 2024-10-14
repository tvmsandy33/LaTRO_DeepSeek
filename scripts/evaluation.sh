BASE_MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
BATCH_SIZE=64
for EVAL_MODEL_NAME in meta-llama/Meta-Llama-3.1-8B-Instruct
do
    for DATASET_NAME in "gsm8k_0shot"
    do
        for RESPONSE_LENGTH in 500
        do
            accelerate launch --config_file configs/evaluation.yaml \
            scripts/evaluation.py \
            --base_model_name_or_path $BASE_MODEL_NAME \
            --eval_model_name_or_path $EVAL_MODEL_NAME \
            --dataset_name $DATASET_NAME \
            --response_length $RESPONSE_LENGTH \
            --temperature 0 \
            --stop_token both \
            --eval_batch_size $BATCH_SIZE \
            --seed 42
        done
    done
done
