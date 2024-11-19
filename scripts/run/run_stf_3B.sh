python scripts/training/run_clm_sft_with_peft.py \
    --model_name_or_path /data/workspace/.cache/huggingface/models--spxiong--Llama-3.2-3B-Chinese-Instruct/snapshots/d98de0fad5669ed16c71467b783c0292c8241d93 \
    --tokenizer_name_or_path /data/workspace/.cache/huggingface/models--spxiong--Llama-3.2-3B-Chinese-Instruct/snapshots/d98de0fad5669ed16c71467b783c0292c8241d93 \
    --dataset_dir data \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train 1 \
    --do_eval 1 \
    --seed 42 \
    --bf16 1 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --max_seq_length 1024 \
    --output_dir lora_output \
    --overwrite_output_dir 1 \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 16 \
    --lora_alpha 128 \
    --trainable "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_dropout 0.05 \
    --modules_to_save "embed_tokens,lm_head" \
    --torch_dtype bfloat16 \
    --validation_file eval/ruozhiba_qa2449_gpt4turbo.json \
    --load_in_kbits 16 \
