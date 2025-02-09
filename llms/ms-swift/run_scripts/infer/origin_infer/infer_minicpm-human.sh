CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type minicpm-v-v2_6-chat \
    --model_id_or_path /data/tfshen/llm/minicpm-v \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/infer/minicpm-v \
    --eval_human True \
    --do_sample False \
