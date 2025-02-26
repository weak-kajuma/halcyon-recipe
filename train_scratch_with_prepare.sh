BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=64
EPOCH=1

CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=smollm2 python ../source/train/run_clm.py \
    --model_type "llama" \
    --model_name_or_path ../scratch_adamw_phase_1/checkpoint-11000 \
    --dataset_name kajuma/training_01-09_token \
    --output_dir ../scratch_adamw_phase_2 \
    --cache_dir ./cache/ \
    --do_train \
    --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --learning_rate 3.0e-3 \
    --weight_decay 0.1 \
    --num_train_epochs $EPOCH \
    --logging_dir ../scratch_adamw_phase_2/logs \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 1000 \
    --eval_steps 500 \
    --save_total_limit 100000 \
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
    --report_to wandb \
    --use_liger_kernel True \
    --seed 42 \
    --optim adamw_bnb_8bit \
    --adam_beta2 0.95 \
    --warmup_steps 1000 \
    --min_lr_rate 0.1 \
    --lr_scheduler_type cosine_with_min_lr \
    --adam_epsilon 1.0e-8
    # --torch_compile True \
    # --torch_compile_backend "eager" \
    # --resume_from_checkpoint ./results/pretrain/pretrain_mistral/trial1/checkpoint-5000/
    # --gradient_checkpointing True 
    # --min_lr 8.0e-6 \
    #--load_best_model_at_end \