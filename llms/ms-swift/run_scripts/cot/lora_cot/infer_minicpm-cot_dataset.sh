# 用于对微调后的模型进行cot指令生成个性化标题

CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --ckpt_dir /home/tfshen/pyproject/pcg/llms/ms-swift/run_scripts/sft/output/minicpm-v-v2_6-chat/v3-20241205-151008/checkpoint-310 \
    --load_dataset_config True \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/lora_cot/minicpm-v \
    --img_dir /data/tfshen/pcg_img_sample \
    --eval_human False \
    --eval_ficl False \
    --eval_udcf False \
    --enforce_cot True \
    --enforce_transfer False \
    --cot_mode dataset \
    --task_mode single \
    --cot_streamline True \
    --use_cot_history_img False \
    --cot_sample_dataset /home/tfshen/pyproject/pcg/data/summary4eval/sample.json \
    --cot_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/lora_cot/minicpm-v \
    --group_index_path /home/tfshen/pyproject/pcg/data/summary4eval/group_index.json \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl
