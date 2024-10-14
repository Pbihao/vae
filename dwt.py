import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as T

# 加载 VAE 模型
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae.eval()

# 读取并预处理图片 (PIL -> Tensor)
image = Image.open("/home/llm/bhpeng/generation/res_vid/vqgan/images/1.JPEG").convert("RGB")
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 512, 512]

# 编码 (从图像到潜在空间)
with torch.no_grad():
    latents = vae.encode(image_tensor).latent_dist.mean

# 打印潜在空间的形状
print(f"Latent space shape: {latents.shape}")

# 解码 (从潜在空间到图像)
with torch.no_grad():
    reconstructed_image = vae.decode(latents).sample

# 后处理 (Tensor -> PIL)
reconstructed_image = (reconstructed_image / 2 + 0.5).clamp(0, 1)  # 反归一化
reconstructed_image = reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy()

# 转换为 PIL Image 并保存
reconstructed_pil_image = Image.fromarray((reconstructed_image * 255).astype("uint8"))
reconstructed_pil_image.save("/home/llm/bhpeng/generation/res_vid/vqgan/images/1c.JPEG")

print("Reconstructed image saved as reconstructed_image.png")
