CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_vqgan.py \
  --image_column=image \
  --validation_images images/1.JPEG images/2.JPEG images/3.JPEG images/4.JPEG \
  --resolution=128 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --report_to=tensorboard \
  --checkpoints_total_limit=1 \
  --train_data_dir=/home/llm/bhpeng/generation/res_vid/vqgan/data/imagenet/train \
  --val_image_dir=vqgan-output/imgs/test


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_vqgan.py \
  --image_column=image \
  --validation_images images/1.JPEG images/2.JPEG images/3.JPEG images/4.JPEG \
  --resolution=256 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=2 \
  --report_to=tensorboard \
  --checkpoints_total_limit=1 \
  --train_data_dir=/home/llm/bhpeng/generation/res_vid/vqgan/data/imagenet/train \
  --val_image_dir=vqgan-output/imgs/test \
  --validation_steps 400 \
  --learning_rate 1e-4 \
  --discr_learning_rate 1e-4



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_vcmodel.py \
  --image_column=image \
  --validation_images images/1.JPEG images/2.JPEG images/3.JPEG images/4.JPEG \
  --resolution=64 \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --report_to=tensorboard \
  --checkpoints_total_limit=1 \
  --train_data_dir=/home/llm/bhpeng/generation/res_vid/vqgan/data/imagenet/train \
  --val_image_dir=vqgan-output/imgs/vcmodel \
  --validation_steps 400 \
  --learning_rate 1e-4 \
  --checkpointing_steps 500 \
  --pretrained_model_name_or_path /home/llm/bhpeng/generation/res_vid/vqgan/vqgan-output/checkpoint-8000/vqmodel


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_vqgan_model.py \
  --image_column=image \
  --validation_images images/1.JPEG images/2.JPEG images/3.JPEG images/4.JPEG \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --report_to=tensorboard \
  --checkpoints_total_limit=1 \
  --train_data_dir=/home/llm/bhpeng/generation/res_vid/vqgan/data/imagenet/val \
  --val_image_dir=vqgan-output/imgs/vcmodel \
  --validation_steps 400 \
  --learning_rate 1e-4 \
  --checkpointing_steps 500 \
  --pretrained_model_name_or_path /home/llm/bhpeng/generation/res_vid/vqgan/vqgan-output/1checkpoint-5000/vqmodel \
  --discriminator_config_name_or_path /home/llm/bhpeng/generation/res_vid/vqgan/vqgan-output/checkpoint-500/discriminator
  

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_vcmodel_vae.py \
  --image_column=image \
  --validation_images images/1.JPEG images/2.JPEG images/3.JPEG images/4.JPEG \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --report_to=tensorboard \
  --checkpoints_total_limit=1 \
  --train_data_dir=/home/llm/bhpeng/generation/res_vid/vqgan/data/imagenet/train \
  --val_image_dir=vqgan-output/imgs/vcmodel \
  --validation_steps 400 \
  --learning_rate 1e-4 \
  --checkpointing_steps 500 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train_cascade_vae.py \
  --image_column=image \
  --validation_images images/1.JPEG images/2.JPEG images/3.JPEG images/4.JPEG \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --report_to=tensorboard \
  --checkpoints_total_limit=1 \
  --train_data_dir=/home/llm/bhpeng/generation/res_vid/vqgan/data/imagenet/train \
  --val_image_dir=vqgan-output/imgs/vcmodel \
  --validation_steps 400 \
  --learning_rate 1e-4 \
  --checkpointing_steps 500 

from model.vcmodel import VQModel
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = VQModel(
            act_fn="silu",
            block_out_channels=[
                32,
                64,
                256,
                1024,
                1024,
                1024,
            ],
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            in_channels=3,
            latent_channels=128,
            layers_per_block=1,
            norm_num_groups=32,
            norm_type="group",
            num_vq_embeddings=16384,
            out_channels=3,
            sample_size=32,
            scaling_factor=0.18215,
            up_block_types=[
                "UpDecoderBlock2D", 
                "UpDecoderBlock2D", 
                "UpDecoderBlock2D", 
                "UpDecoderBlock2D", 
                "UpDecoderBlock2D", 
                "UpDecoderBlock2D"],
            vq_embed_dim=4,
        )