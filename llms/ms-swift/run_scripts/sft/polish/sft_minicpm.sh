# 目前用于训练润色模型

CUDA_VISIBLE_DEVICES=1 swift sft \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path /data/tfshen/llm/minicpm-v \
    --output_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/sft/polish/minicpm-v \
    --num_train_epochs 5 \
    --sft_type lora \
    --gradient_accumulation_steps 1 \
    --dataset /home/tfshen/pyproject/pcg/data/refinement/train.jsonl \
