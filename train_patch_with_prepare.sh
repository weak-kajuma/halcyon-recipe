BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=32
EPOCH=1

python run_clm_patch.py \
    --model_type "llama" \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --dataset_name kajuma/training_01-09_patch \
    --patch_size 4 \
    --output_dir ../results \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --prediction_loss_only \
    --remove_unused_columns False \
    --learning_rate 3.0e-3 \
    --weight_decay 0.1 \
    --num_train_epochs $EPOCH \
    --logging_dir ../results  \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 1000 \
    --eval_steps 1000 \
    --save_total_limit 1000000 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --low_cpu_mem_usage \
    --block_size 1024 \
    --torch_dtype "bfloat16" \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --push_to_hub False \
    --preprocessing_num_workers 12 \
    --dataloader_num_workers 12 \
    --attn_implementation "flash_attention_2" \
    --use_liger_kernel True \
    --report_to none \
    --seed 42 \
    --optim schedule_free_radam \
    --adam_beta2 0.95 \
    --lr_scheduler_type constant \
    --adam_epsilon 1.0e-8
    # --config_name ../source/model/config/config_llama_300m.json \
    # --resume_from_checkpoint ~/checkpoint-5000/
    # --torch_compile True \
    # --torch_compile_backend "eager" \
    # --resume_from_checkpoint ./results/pretrain/pretrain_mistral/trial1/checkpoint-5000/
    # --gradient_checkpointing True 
    # --min_lr 8.0e-6 \
    #--load_best_model_at_end \