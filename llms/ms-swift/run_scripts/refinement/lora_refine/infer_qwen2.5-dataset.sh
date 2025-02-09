# 对minicpm-v生成的标题文件进行润色(使用LoRA Qwen2.5)

CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --ckpt_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/sft/polish/Qwen2_5-7B-Instruct/qwen2_5-7b-instruct/v2-20250105-213545/checkpoint-60 \
    --load_dataset_config True \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/refinement/lora_refine/Qwen2.5 \
    --eval_human False \
    --enforce_refinement True \
    --refinement_mode lora-dataset \
    --refinement_inherient_file /home/tfshen/pyproject/pcg/data/summary4aba/Inherient/qwen2-vl_all-use_cot_new.json \
    --refinement_sample_dataset /home/tfshen/pyproject/pcg/data/summary4aba/sample.json \
    --refinement_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/refinement/lora_refine/Qwen2.5 \
    --infer_backend pt \
    --stream True \
    --do_sample False \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl