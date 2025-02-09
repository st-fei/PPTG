CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type mplug-owl3-7b-chat \
    --model_id_or_path /data/tfshen/llm/mplug-owl3 \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/infer/mplug-owl3 \
    --eval_human False \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl
