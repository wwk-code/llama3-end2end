import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B"

# 指定参数
kwargs = {
    'max_new_tokens': 96,
    'eos_token_id': None,
    'pad_token_id': 1  # 根据需要设置
}

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

output = pipe("The key to life is", **kwargs)
print(output)
