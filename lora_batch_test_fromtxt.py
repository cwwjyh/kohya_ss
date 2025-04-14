import torch
from diffusers import FluxPipeline
import os

# 基础设置
pipe_id = "black-forest-labs/FLUX.1-dev"
cache_dir = '/home/i-caiweiwei/.cache/huggingface/hub'

lora_weight = "/data/caiweiwei/kohya_ss/outputs/objaverse_8gpu/"
lora_name= "/data/caiweiwei/kohya_ss/outputs/objaverse_8gpu/objaverse_8gpu-step00030000.safetensors"

output_dir = "/data/caiweiwei/kohya_ss/outputs/inference/objaverse_test/testjson"
os.makedirs(output_dir, exist_ok=True)

# 加载模型
pipe = FluxPipeline.from_pretrained(pipe_id, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
print("Loading Lora weights")
pipe.load_lora_weights(lora_weight, weight_name=lora_name)
pipe.fuse_lora(lora_scale=1.0)
pipe.to("cuda")

# 图像设置
image_width = 512
image_height = 512

# 从txt文件中读取prompts
prompts_file_path = "/data/caiweiwei/kohya_ss/data/objaverse_testjson.txt"  # 请替换为实际的txt文件路径
prompts = []
with open(prompts_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        prompts.append(line.strip())

# 为每个prompt生成图像
for idx, prompt in enumerate(prompts):
    # 生成随机种子
    seed = torch.randint(0, 100000, (1,), device="cuda")[0].item()
    print(f"Processing prompt {idx+1}/{len(prompts)}")
    print(f"Using seed: {seed}")
    print("Generating image with prompt:", prompt)

    images = pipe(
        prompt,
        height=image_height,
        width=image_width,
        guidance_scale=3.5,
        num_inference_steps=20,
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(seed),
        num_images_per_prompt=1,
    ).images

    # 保存结果，使用索引命名
    output_path = os.path.join(output_dir, f"{idx+1}_30000ckp.png")
    images[0].save(output_path)
    print(f"Image saved to: {output_path}")

print("所有图片生成完成！")