python prepare_dataset.py \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --dataset_name kajuma/ABEJA-CC-JA-edu \
    --dataset_config_name "30%" \
    --text_column_name content \
    --output_dir /mnt/raid0/data \
    --block_size 2048 \
    --preprocessing_num_workers 24 \
    --save_num_workers 24 \
    --seed 3407