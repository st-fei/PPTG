# 对minicpm-v生成的标题文件进行润色(使用minicpm-v)

CUDA_VISIBLE_DEVICES=1 swift infer \
    --ckpt_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/sft/polish/minicpm-v/minicpm-v-v2_6-chat/v0-20250107-150249/checkpoint-60 \
    --load_dataset_config True \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/refinement/lora_refine/MiniCPM-V \
    --eval_human False \
    --enforce_refinement True \
    --refinement_mode lora-dataset \
    --refinement_inherient_file /home/tfshen/pyproject/pcg/data/summary4aba/Inherient/minicpm-v_all-use_cot.json \
    --refinement_sample_dataset /home/tfshen/pyproject/pcg/data/summary4aba/sample.json \
    --refinement_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/refinement/lora_refine/MiniCPM-V \
    --infer_backend pt \
    --stream True \
    --do_sample False \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl