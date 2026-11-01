##### Question rewrite

```
python src/main.py \
--d MultiTQ \
--triplet_path /path/to/datasets/MultiTQ/kg/full.txt \
--question_path /path/to/datasets/MultiTQ/test.json \
--rewrite_output_path /path/to/MultiTQ/MultiTQ_rewrite.json 
```



##### Knowledge fine-tuning

```
python -u LLMs/LLaMA/src/train_bash.py \
--stage sft \
--model_name_or_path /path/to/Llama-2-7b-hf \
--do_train  \
--dataset_dir LLMs/data_id \
--dataset ${data}_train \
--template default  \
--finetuning_type lora \
--lora_target gate_proj,down_proj,up_proj \
--output_dir "${checkpoint_dir}" \
--overwrite_cache \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4  \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_strategy no \
--learning_rate 5e-5  \
--num_train_epochs 2 \
--overwrite_output_dir \
--topk 25 \
--soft_prompt_length 32 \
--extra_infor_len 32 \
--gate_len 16 
```



```
python -u LLMs/LLaMA/src/eval.py  \
--model_name_or_path  /path/to/Llama-2-7b-hf   \
--dataset_dir LLMs/data_id  \
--dataset ${data}_test  \
--input_file LLMs/data_id/MultiTQ_test/MultiTQ_rewrite.json \
--template default  \
--finetuning_type lora  \
--lora_target gate_proj,down_proj,up_proj  \
--checkpoint_dir "${checkpoint_dir}"  \
--max_new_tokens 256 \
--topk 25 \
--soft_prompt_length 32 \
--extra_infor_len 32 \
--gate_len 16
```

