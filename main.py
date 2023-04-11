import torch
from kandinsky2 import get_kandinsky2

model = get_kandinsky2(
    'cuda', 
    task_type='text2img', 
    cache_dir='/tmp/kandinsky2', 
    model_version='2.1', 
    use_flash_attention=False
)

def torch_gc():
   with torch.cuda.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
