CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type qwen2-vl-7b-instruct \
    --model_id_or_path /data/tfshen/llm/qwen2-vl \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/infer/qwen2-vl \
    --eval_human True \
