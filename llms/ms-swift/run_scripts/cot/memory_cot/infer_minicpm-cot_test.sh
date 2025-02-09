# 用于跑memory mode的test

CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path /data/tfshen/llm/minicpm-v \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/cot/memory_cot/minicpm-v \
    --img_dir /data/tfshen/sft_img_sample \
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
