# 用于执行format-coT的dataset模式
# 1.3 被用来跑训练集
# 原始数据如下：
# --cot_sample_dataset /home/tfshen/pyproject/pcg/data/summary4eval/sample.json \
# --img_dir /data/tfshen/pcg_img_sample \
# --is_sample True \

CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path /data/tfshen/llm/minicpm-v \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/origin_cot/minicpm-v \
    --img_dir /data/tfshen/sft_img_sample \
    --is_sample False \
    --eval_human False \
    --eval_ficl False \
    --eval_udcf False \
    --enforce_cot True \
    --enforce_transfer False \
    --cot_mode dataset \
    --cot_streamline True \
    --use_cot_history_img True \
    --cot_sample_dataset /home/tfshen/pyproject/pcg/data/sft/sft_data/large/sft_train.json \
    --cot_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/origin_cot/minicpm-v \
    --group_index_path /home/tfshen/pyproject/pcg/data/summary4eval/group_index.json \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl
