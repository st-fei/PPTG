export CUDA_VISIBLE_DEVICES=0,1


# 执行推理命令，将当前值传递给相应的参数
for EXAMPLE_NUM in 0 3 5; do
    for USE_HISTORY_TITLE in False True; do
        for USE_HISTORY_IMG in False True; do
            if [ "$USE_HISTORY_TITLE" = "False" ] && [ "$USE_HISTORY_IMG" = "True" ]; then
                continue
            fi
            for USE_TARGET_IMG in False True; do
                echo "Now running $EXAMPLE_NUM-$USE_HISTORY_TITLE-$USE_HISTORY_IMG-$USE_TARGET_IMG"
                swift infer \
                    --model_type minicpm-v-v2_6-chat \
                    --model_id_or_path  /data/tfshen/llm/minicpm-v \
                    --result_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/ficl/minicpm-v \
                    --eval_human False \
                    --eval_ficl True \
                    --ficl_example_num $EXAMPLE_NUM \
                    --use_ficl_history_title $USE_HISTORY_TITLE \
                    --use_ficl_history_img $USE_HISTORY_IMG \
                    --use_ficl_target_img $USE_TARGET_IMG \
                    --ficl_model_gen_path /home/tfshen/pyproject/pcg/data/summary4eval/model_sample_gen.json \
                    --ficl_sample_dataset /home/tfshen/pyproject/pcg/data/summary4eval/sample.json \
                    --ficl_save_dir /home/tfshen/pyproject/pcg/llms/ms-swift/result/ficl/minicpm-v \
                    --val_dataset /home/tfshen/pyproject/pcg/llms/ms-swift/dataset/pcg/processed/add.jsonl
            done
        done
    done
done
