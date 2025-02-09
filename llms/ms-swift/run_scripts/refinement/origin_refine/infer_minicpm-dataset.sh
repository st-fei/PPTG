# 对minicpm-v生成的标题文件进行润色

CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type qwen2_5-7b-instruct \
    --model_id_or_path /data/LLM/Qwen2.5-7B-Instruct \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/refinement/origin_refine/minicpm-v \
    --eval_human False \
    --enforce_refinement True \
    --refinement_mode dataset \
    --refinement_inherient_file /home/tfshen/pyproject/pcg/data/summary4aba/Inherient/minicpm-v_all-use_cot.json \
    --refinement_sample_dataset /home/tfshen/pyproject/pcg/data/summary4aba/sample.json \
    --refinement_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/refinement/origin_refine/minicpm-v \
    --infer_backend pt \
    --stream True \
    --do_sample False \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl