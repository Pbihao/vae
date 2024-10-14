# import torch

# def dct_2d(block):
#     N = 8
#     dct_block = torch.zeros((N, N))

#     x = torch.arange(N).float()
#     cosines = torch.cos((2 * x.unsqueeze(1) + 1) * x * 3.141592653589793 / (2 * N))  # (8, 8)

#     # 计算 DCT
#     for u in range(N):
#         for v in range(N):
#             C_u = 1 / (2 ** 0.5) if u == 0 else 1
#             C_v = 1 / (2 ** 0.5) if v == 0 else 1
            
#             dct_block[u, v] = (1 / 4) * C_u * C_v * torch.sum(block * cosines[u, v])

#     return dct_block

# def compute_dct_loss(tensor1, tensor2):
#     B, C, H, W = tensor1.shape
#     assert tensor1.shape == tensor2.shape, "输入的两个张量必须具有相同的形状"

#     blocks1 = tensor1.unfold(2, 8, 8).unfold(3, 8, 8)  # (B, C, num_blocks, 8, 8)
#     blocks2 = tensor2.unfold(2, 8, 8).unfold(3, 8, 8)  # (B, C, num_blocks, 8, 8)

#     # 计算 DCT
#     dct_blocks1 = torch.empty(B, C, blocks1.size(2), 8, 8).to(tensor1)
#     dct_blocks2 = torch.empty(B, C, blocks2.size(2), 8, 8).to(tensor2)

#     for b in range(B):
#         for c in range(C):
#             for block_idx in range(blocks1.size(2)):
#                 block1 = blocks1[b, c, block_idx]  # 取出当前块
#                 block2 = blocks2[b, c, block_idx]  # 取出当前块
                
#                 # 计算 DCT
#                 dct_blocks1[b, c, block_idx] = dct_2d(block1)
#                 dct_blocks2[b, c, block_idx] = dct_2d(block2)

#     # 计算 DCT 之间的差异
#     loss = torch.mean((dct_blocks1 - dct_blocks2) ** 2)

#     return loss

# # # 示例用法
# # tensor1 = torch.rand((1, 3, 256, 256))  # Batch size 1, 3 channels, 64x64 image
# # tensor2 = torch.rand((1, 3, 256, 256))  # Batch size 1, 3 channels, 64x64 image

# # loss_value = compute_dct_loss(tensor1, tensor2)
# # print("Loss:", loss_value.item())




import torch

def dct_2d(blocks):
    """计算 2D 离散余弦变换（DCT），支持批量输入"""
    N = 8
    # 计算 DCT 系数
    x = torch.arange(N, dtype=torch.float32).unsqueeze(1).to(blocks)  # (8, 1)
    y = torch.arange(N, dtype=torch.float32).unsqueeze(0).to(blocks)  # (1, 8)
    
    # 预计算余弦值
    cosines = torch.cos((2 * x + 1) * y * 3.141592653589793 / (2 * N))  # (8, 8)
    
    # 计算 DCT 系数
    C_u = 1 / (2 ** 0.5) * (x == 0).float() + 1  # (8,)
    C_v = 1 / (2 ** 0.5) * (y == 0).float() + 1  # (8,)
    
    # 计算 DCT
    dct_block = (C_u.view(1, N, 1) * C_v.view(1, 1, N) / 4) * (blocks.unsqueeze(2).unsqueeze(3) * cosines).sum(dim=(-1, -2))
    
    return dct_block  # 返回的形状是 (B*C, 8, 8)

def compute_dct_loss(tensor1, tensor2):
    """计算两个张量在频域中的差距"""
    B, C, H, W = tensor1.shape
    assert tensor1.shape == tensor2.shape, "输入的两个张量必须具有相同的形状"

    # 使用 unfold 来提取 8x8 块
    blocks1 = tensor1.unfold(2, 8, 8).unfold(3, 8, 8)  # (B, C, num_blocks, 8, 8)
    blocks2 = tensor2.unfold(2, 8, 8).unfold(3, 8, 8)  # (B, C, num_blocks, 8, 8)

    # 重新调整形状以支持批量处理
    blocks1 = blocks1.reshape(B * C, -1, 8, 8)  # (B*C, num_blocks, 8, 8)
    blocks2 = blocks2.reshape(B * C, -1, 8, 8)  # (B*C, num_blocks, 8, 8)

    # 计算 DCT
    dct_blocks1 = dct_2d(blocks1.view(-1, 8, 8))  # 计算 DCT
    dct_blocks2 = dct_2d(blocks2.view(-1, 8, 8))  # 计算 DCT

    # 重新调整形状
    dct_blocks1 = dct_blocks1.view(B, C, -1, 8, 8)
    dct_blocks2 = dct_blocks2.view(B, C, -1, 8, 8)

    # 计算 DCT 之间的差异
    loss = torch.mean((dct_blocks1 - dct_blocks2) ** 2)

    return loss

# # 示例用法
# tensor1 = torch.rand((1, 3, 256, 256),)  # Batch size 1, 3 channels, 64x64 image
# tensor2 = torch.rand((1, 3, 256, 256),)  # Batch size 1, 3 channels, 64x64 image

# loss_value = compute_loss(tensor1, tensor2)
# print("Loss:", loss_value.item())