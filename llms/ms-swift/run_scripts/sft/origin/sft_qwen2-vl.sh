# origin sft for qwen2-vl

CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type qwen2-vl-7b-instruct \
    --model_id_or_path /data/tfshen/llm/qwen2-vl \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/sft/origin \
    --eval_steps 500 \
    --gradient_accumulation_steps 16 \
    --dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/sft/large/stack/sft_train.jsonl \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/sft/large/stack/sft_dev.jsonl