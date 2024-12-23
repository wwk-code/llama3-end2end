huggingface_model_path='/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08'
# huggingface_model_path='/data/workspace/.cache/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08'

python scripts/training/run_clm_sft_with_peft.py \
  --model_name_or_path ${huggingface_model_path} \
  --tokenizer_name_or_path ${huggingface_model_path} \
  --dataset_dir data \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --do_train 1 \
  --do_eval 1 \
  --seed 42 \
  --bf16 1 \
  --num_train_epochs 2 \
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
  --max_seq_length 512 \
  --output_dir lora_output \
  --overwrite_output_dir 1 \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --lora_rank 32 \
  --lora_alpha 128 \
  --trainable "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
  --lora_dropout 0.05 \
  --modules_to_save "embed_tokens,lm_head" \
  --torch_dtype bfloat16 \
  --validation_file eval/ruozhiba_qa2449_gpt4turbo.json \
  --load_in_kbits 16
