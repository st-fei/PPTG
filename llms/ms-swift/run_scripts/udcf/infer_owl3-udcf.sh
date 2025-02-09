CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type mplug-owl3-7b-chat \
    --model_id_or_path /data/tfshen/llm/mplug-owl3 \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/udcf/mplug-owl3 \
    --eval_human False \
    --eval_ficl False \
    --eval_udcf True \
    --udcf_example_num 0 \
    --udcf_rank_num 4 \
    --enable_udcf_cot False \
    --udcf_model_gen_path /home/tfshen/pyproject/pcg/data/summary4eval/model_sample_gen.json \
    --udcf_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/udcf/mplug-owl3 \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl
    # --udcf_sample_dataset /home/tfshen/pyproject/pcg/data/summary4eval/sample.json \

