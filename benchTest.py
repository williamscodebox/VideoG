import torch, time
from diffusers import DiffusionPipeline

t0 = time.time()
pipe = DiffusionPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
print("Load time:", time.time() - t0)

t1 = time.time()
with torch.autocast("cuda"):
    out = pipe(prompt="test", num_frames=8)
print("Inference time:", time.time() - t1)
