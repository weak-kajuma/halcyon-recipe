BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=32
EPOCH=1

CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=halcyon python run_clm_patch_with_prepare.py \
    --model_type "llama" \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --preprocessed_dataset_name kajuma/training_02_26 \
    --patch_size 4 \
    --output_dir ./patch \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --num_train_epochs $EPOCH \
    --logging_dir ./patch/logs  \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 300 \
    --eval_steps 300 \
    --save_total_limit 1000000 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --low_cpu_mem_usage \
    --block_size 1024 \
    --torch_dtype "bfloat16" \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --push_to_hub False \
    --preprocessing_num_workers 24 \
    --dataloader_num_workers 24 \
    --attn_implementation "flash_attention_2" \
    --use_liger_kernel True \
    --report_to wandb \
    --seed 3407 \
    --optim schedule_free_radam \
    --learning_rate 5.0e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --lr_scheduler_type constant \
    --adam_epsilon 1.0e-8
    # --resume_from_checkpoint ./scratch/checkpoint-5000/
