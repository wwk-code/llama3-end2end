# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]

kwargs = {'max_new_tokens': 96}

pipe = pipeline("text-generation", model="spxiong/Llama-3.2-3B-Chinese-Instruct",device='cuda:0')
output = pipe(messages, **kwargs)

print(output)
