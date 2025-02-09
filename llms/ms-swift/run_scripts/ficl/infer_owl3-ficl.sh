# 执行推理命令，将当前值传递给相应的参数
CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --model_type mplug-owl3-7b-chat \
    --model_id_or_path /data/tfshen/llm/mplug-owl3 \
    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/ficl/mplug-owl3 \
    --eval_human False \
    --eval_ficl True \
    --ficl_example_num 5 \
    --use_ficl_history_title True \
    --use_ficl_history_img True \
    --use_ficl_target_img True \
    --ficl_model_gen_path /home/tfshen/pyproject/pcg/data/summary4eval/model_sample_gen.json \
    --ficl_sample_dataset /home/tfshen/pyproject/pcg/data/summary4eval/sample.json \
    --ficl_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/ficl/mplug-owl3 \
    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl
            
