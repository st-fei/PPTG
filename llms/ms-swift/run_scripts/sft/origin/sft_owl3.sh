# origin sft for mplug-owl3

CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type mplug-owl3-7b-chat \
    --model_id_or_path /data/tfshen/llm/mplug-owl3 \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/sft/origin \
    --eval_steps 1000 \
    --gradient_accumulation_steps 4 \
    --dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/sft/large/stack/sft_train.jsonl \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/sft/large/stack/sft_dev.jsonl
