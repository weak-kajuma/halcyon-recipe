# global batch size = 512
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=512

CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=halcyon accelerate launch run_clm.py \
    --model_type "llama" \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --preprocessed_dataset_name kajuma/training_03_05 \
    --streaming True \
    --output_dir ./scratch \
    --do_train \
    --prediction_loss_only \
    --remove_unused_columns False \
    --max_steps 20480 \
    --logging_dir ./scratch/logs \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --per_device_train_batch_size $BATCH_SIZE \
    --low_cpu_mem_usage \
    --torch_dtype "bfloat16" \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --push_to_hub True \
    --hub_strategy every_save \
    --hub_model_id halcyon-llm/Llama-halcyon-1B-scratch \
    --dataloader_num_workers 24 \
    --attn_implementation "flash_attention_2" \
    --report_to wandb \
    --use_liger_kernel True \
    --seed 3407 \
    --optim adamw_bnb_8bit \
    --warmup_steps 300 \
    --min_lr_rate 0.1 \
    --learning_rate 5.0e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --lr_scheduler_type cosine_with_min_lr \
    --adam_epsilon 1.0e-8
    # --resume_from_checkpoint ./scratch/checkpoint-5000/
