# 用于为format-coT生成的标题进行dataset mode的润色

CUDA_VISIBLE_DEVICES=1 swift infer \
    --ckpt_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/sft/polish/minicpm-v-v2_6-chat/v0-20241209-164951/checkpoint-75 \
    --load_dataset_config True \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/memory_cot/qwen2-vl \
    --inherient_gen_file /home/tfshen/pyproject/pcg/data/summary4aba/Inherient/qwen2-vl_wo-history_cot.json \
    --eval_human False \
    --eval_ficl False \
    --eval_udcf False \
    --enforce_cot True \
    --cot_mode dataset \
    --control_memory_bank True \
    --cot_streamline True \
    --use_cot_history_img False \
    --enable_visible_img True \
    --enable_hidden_img False \
    --cot_sample_dataset /home/tfshen/pyproject/pcg/data/summary4eval/sample.json \
    --cot_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/memory_cot/qwen2-vl \
    --group_index_path /home/tfshen/pyproject/pcg/data/summary4eval/group_index.json \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl




