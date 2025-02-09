CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type qwen2_5-7b-instruct \
    --model_id_or_path /data/LLM/Qwen2.5-7B-Instruct \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/infer/origin_infer/qwen2.5-7b-instruct \
    --eval_human True \
    --infer_backend pt \
    --stream True \
    --do_sample False \