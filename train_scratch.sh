# global batch size = 2048
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=64
EPOCH=1

CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=halcyon python run_clm.py \
    --model_type "llama" \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --preprocessed_dataset_name kajuma/training_01-09_token \
    --output_dir ./scratch \
    --cache_dir ./cache/ \
    --do_train \
    --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --num_train_epochs $EPOCH \
    --logging_dir ./scratch/logs \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 300 \
    --eval_steps 300 \
    --save_total_limit 100000 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --low_cpu_mem_usage \
    --torch_dtype "bfloat16" \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --push_to_hub False \
    --preprocessing_num_workers 24 \
    --dataloader_num_workers 24 \
    --attn_implementation "flash_attention_2" \
    --report_to wandb \
    --use_liger_kernel True \
    --seed 3407 \
    --optim schedule_free_radam \
    --learning_rate 3.0e-3 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --lr_scheduler_type constant \
    --adam_epsilon 1.0e-8
    # --resume_from_checkpoint ./scratch/checkpoint-5000/
