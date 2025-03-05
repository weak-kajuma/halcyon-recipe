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
    --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --max_steps 20480 \
    --logging_dir ./scratch/logs \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 1000 \
    --eval_steps 1000 \
    --save_total_limit 100000 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --low_cpu_mem_usage \
    --torch_dtype "bfloat16" \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --push_to_hub True \
    --hub_strategy all_checkpoints \
    --hub_model_id halcyon-llm/test \
    --dataloader_num_workers 24 \
    --attn_implementation "flash_attention_2" \
    --report_to wandb \
    --use_liger_kernel False \
    --seed 3407 \
    --optim schedule_free_radam \
    --learning_rate 3.0e-3 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --lr_scheduler_type constant \
    --adam_epsilon 1.0e-8
    # --resume_from_checkpoint ./scratch/checkpoint-5000/
