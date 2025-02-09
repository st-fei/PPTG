# 目前用于训练润色模型

CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type qwen2_5-7b-instruct \
    --model_id_or_path /data/LLM/Qwen2.5-7B-Instruct \
    --output_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/sft/polish/Qwen2_5-7B-Instruct \
    --num_train_epochs 5 \
    --sft_type lora \
    --gradient_accumulation_steps 1 \
    --dataset /home/tfshen/pyproject/pcg/data/refinement/train.jsonl \
