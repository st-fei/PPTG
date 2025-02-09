# 用于执行format-coT的dataset模式

CUDA_VISIBLE_DEVICES=1 swift infer \
    --model_type qwen2-vl-7b-instruct \
    --model_id_or_path /data/tfshen/llm/qwen2-vl \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/origin_cot/qwen2-vl \
    --img_dir /data/tfshen/pcg_img_sample \
    --eval_human False \
    --eval_ficl False \
    --eval_udcf False \
    --enforce_cot True \
    --enforce_transfer False \
    --cot_mode dataset \
    --cot_streamline True \
    --use_cot_history_img True \
    --cot_sample_dataset /home/tfshen/pyproject/pcg/data/summary4eval/sample.json \
    --cot_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/origin_cot/qwen2-vl \
    --group_index_path /home/tfshen/pyproject/pcg/data/summary4eval/group_index.json \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl
