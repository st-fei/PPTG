# 用于直接微调

CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path /data/tfshen/llm/minicpm-v \
    --output_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/sft/origin/minicpm-v \
    --num_train_epochs 3 \
    --eval_steps 500 \
    --sft_type lora \
    --dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/sft/large/stack/sft_train.jsonl \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/sft/large/stack/sft_dev.jsonl \
