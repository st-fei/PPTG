export RESIZED_HEIGHT=512  # 设置一个较小的高度
export RESIZED_WIDTH=512   # 设置一个较小的宽度
CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type qwen2-vl-7b-instruct \
    --model_id_or_path /data/tfshen/llm/qwen2-vl \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/infer/qwen2-vl \
    --eval_human False \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl
