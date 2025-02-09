# 用于为format-coT生成的标题进行test mode的润色

CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --ckpt_dir /home/tfshen/pyproject/pcg/llms/ms-swift/run_scripts/sft/output/minicpm-v-v2_6-chat/v1-20241209-163049/checkpoint-35 \
    --load_dataset_config True \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/memory_cot/minicpm-v \
    --inherient_gen_file /home/tfshen/pyproject/pcg/data/summary4aba/Inherient/minicpm-v_all-use.json \
    --eval_human False \
    --eval_ficl False \
    --eval_udcf False \
    --enforce_cot True \
    --cot_mode test \
    --control_memory_bank True \
    --cot_streamline True \
    --use_cot_history_img False \
    --enable_visible_img True \
    --enable_hidden_img False \
    --cot_sample_dataset /home/tfshen/pyproject/pcg/data/summary4eval/sample.json \
    --cot_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/memory_cot/minicpm-v \
    --group_index_path /home/tfshen/pyproject/pcg/data/summary4eval/group_index.json \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl




