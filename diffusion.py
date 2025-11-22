import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
from torchvision.utils import make_grid
import argparse
import json
import pandas as pd
from datetime import datetime
import os
# ==================== 基础组件定义 ====================

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + self.res_conv(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.reshape(b, self.heads, -1, h * w), qkv
        )
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: t.reshape(b, self.heads, -1, h * w), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
    ):
        super().__init__()
        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

# ==================== 噪声调度定义 ====================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

# ==================== 扩散模型类 ====================

class DiffusionModel:
    def __init__(self, timesteps=1000, beta_schedule='linear', img_size=32, channels=1):
        self.timesteps = timesteps
        self.img_size = img_size
        self.channels = channels
        
        # 选择噪声调度
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 前向过程计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # 反向过程后验方差
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
            
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # 使用模型预测噪声
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, device):
        b = shape[0]
        # 从纯噪声开始
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, device='cuda'):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), device=device)
    
    # DDIM采样方法
    @torch.no_grad()
    def ddim_sample(self, model, shape, device, ddim_timesteps=50, eta=0.0):
        b = shape[0]
        # 创建DDIM的时间步序列
        times = torch.linspace(-1, self.timesteps - 1, steps=ddim_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        img = torch.randn(shape, device=device)
        imgs = [img.cpu().numpy()]
        
        for time, time_next in tqdm(time_pairs, desc='DDIM sampling'):
            time_cond = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise = model(img, time_cond)
            
            # 计算alpha和sigma
            alpha_prod_t = self.extract(self.alphas_cumprod, time_cond, img.shape)
            alpha_prod_t_next = self.extract(self.alphas_cumprod, 
                                            torch.full((b,), time_next, device=device, dtype=torch.long), 
                                            img.shape) if time_next >= 0 else torch.ones_like(alpha_prod_t)
            
            sigma_t = eta * torch.sqrt((1 - alpha_prod_t_next) / (1 - alpha_prod_t)) * torch.sqrt(1 - alpha_prod_t / alpha_prod_t_next)
            
            # 预测x0
            pred_x0 = (img - torch.sqrt(1 - alpha_prod_t) * pred_noise) / torch.sqrt(alpha_prod_t)
            
            # 方向指向xt
            dir_xt = torch.sqrt(1 - alpha_prod_t_next - sigma_t**2) * pred_noise
            
            if time_next < 0:
                img = pred_x0
            else:
                noise = torch.randn_like(img) if eta > 0 else 0
                img = torch.sqrt(alpha_prod_t_next) * pred_x0 + dir_xt + sigma_t * noise
            
            imgs.append(img.cpu().numpy())
        
        return imgs

# ==================== FID计算工具 ====================

def calculate_fid(real_images, generated_images):
    """
    计算Fréchet Inception Distance (FID)
    简化版本，实际应用中应使用预训练的Inception网络
    """
    # 将图像转换为特征向量（这里使用简单的统计特征作为示例）
    def get_features(images):
        # 在实际应用中，这里应该使用预训练的InceptionV3网络提取特征
        # 这里简化为使用图像的均值和标准差
        features = []
        for img in images:
            if len(img.shape) == 3:
                img = img.mean(axis=0)  # 转为灰度
            features.append([img.mean(), img.std()])
        return np.array(features)
    
    real_features = get_features(real_images)
    gen_features = get_features(generated_images)
    
    # 计算均值和协方差
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    
    # 计算FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# ==================== 训练和实验函数 ====================

def train_model(config):
    """训练扩散模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 初始化模型和扩散过程
    model = Unet(
        dim=config.model_dim,
        channels=1,
        dim_mults=(1, 2, 4, 8)
    ).to(device)
    
    diffusion = DiffusionModel(
        timesteps=config.timesteps,
        beta_schedule=config.beta_schedule,
        img_size=config.img_size,
        channels=1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # 训练循环
    losses = []
    model.train()
    
    for epoch in range(config.epochs):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.epochs}')):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # 随机采样时间步
            t = torch.randint(0, config.timesteps, (batch_size,), device=device).long()
            
            # 计算损失
            loss = diffusion.p_losses(model, data, t, loss_type="huber")
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    return model, diffusion, losses





# ==================== 实验配置 ====================

class Config:
    def __init__(self, timesteps=200, beta_schedule='linear', sampling_method='both', 
                 model_dim=64, img_size=32, batch_size=128, lr=1e-4, epochs=20):
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.sampling_method = sampling_method
        self.model_dim = model_dim
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
    
    def __repr__(self):
        return f"Config(timesteps={self.timesteps}, beta_schedule='{self.beta_schedule}', sampling_method='{self.sampling_method}')"
# 修改run_experiment函数以接受保存目录参数
def run_experiment(config, experiment_name, save_dir=None):
    """运行单个实验"""
    if save_dir is None:
        save_dir = "."
    
    print(f"\n=== 运行实验: {experiment_name} ===")
    print(f"配置: {config}")
    
    # 训练模型
    model, diffusion, losses = train_model(config)
    
    # 生成样本
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # DDPM采样
    if config.sampling_method == 'ddpm' or config.sampling_method == 'both':
        print("使用DDPM采样...")
        ddpm_samples = diffusion.sample(model, image_size=config.img_size, batch_size=16, channels=1, device=device)
        ddpm_final = ddpm_samples[-1]
    
    # DDIM采样
    if config.sampling_method == 'ddim' or config.sampling_method == 'both':
        print("使用DDIM采样...")
        ddim_samples = diffusion.ddim_sample(model, shape=(16, 1, config.img_size, config.img_size), 
                                           device=device, ddim_timesteps=50)
        ddim_final = ddim_samples[-1]
    
    # 计算FID（简化版）
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)
    real_images = next(iter(test_dataloader))[0].numpy()
    
    fid_scores = {}
    if config.sampling_method == 'ddpm' or config.sampling_method == 'both':
        fid_scores['ddpm'] = calculate_fid(real_images, ddpm_final)
    
    if config.sampling_method == 'ddim' or config.sampling_method == 'both':
        fid_scores['ddim'] = calculate_fid(real_images, ddim_final)
    
    # 保存结果
    results = {
        'config': config,
        'losses': losses,
        'fid_scores': fid_scores,
        'experiment_name': experiment_name
    }
    
    # 可视化一些样本
    if config.sampling_method == 'ddpm' or config.sampling_method == 'both':
        plt.figure(figsize=(10, 10))
        grid = make_grid(torch.tensor(ddpm_final[:16]), nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"DDPM Samples - {experiment_name}")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{experiment_name}_ddpm_samples.png'))
        plt.close()
    
    if config.sampling_method == 'ddim' or config.sampling_method == 'both':
        plt.figure(figsize=(10, 10))
        grid = make_grid(torch.tensor(ddim_final[:16]), nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"DDIM Samples - {experiment_name}")
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{experiment_name}_ddim_samples.png'))
        plt.close()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f'Training Loss - {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, f'{experiment_name}_loss.png'))
    plt.close()
    
    return results

# 修改demonstrate_latent_quality函数以接受保存目录
def demonstrate_latent_quality(best_model, best_diffusion, config, save_dir):
    """展示潜在向量质量对生成图像的影响"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.eval()
    
    print("\n=== 潜在向量质量实验 ===")
    
    # 测试不同尺度的噪声
    scales = [0.5, 1.0, 2.0, 4.0]  # 噪声标准差
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, scale in enumerate(scales):
        # 生成不同尺度的噪声
        noise = torch.randn(8, 1, config.img_size, config.img_size, device=device) * scale
        
        # 使用DDIM快速采样
        samples = best_diffusion.ddim_sample(
            best_model, 
            shape=noise.shape, 
            device=device,
            ddim_timesteps=50
        )
        
        final_images = samples[-1]
        
        # 显示结果
        for j in range(2):
            ax = axes[j, i]
            if j == 0:
                # 显示噪声
                noise_vis = noise[j].cpu().squeeze().numpy()
                ax.imshow(noise_vis, cmap='gray')
                ax.set_title(f'Noise (scale={scale})')
            else:
                # 显示生成的图像
                img_vis = final_images[j].squeeze()
                ax.imshow(img_vis, cmap='gray')
                ax.set_title(f'Generated (scale={scale})')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latent_quality_experiment.png'))
    plt.close()
    
    # 保存潜在质量实验的描述
    desc_path = os.path.join(save_dir, 'experiment_description.txt')
    with open(desc_path, 'w') as f:
        f.write("潜在向量质量实验\n")
        f.write("=" * 30 + "\n\n")
        f.write("这个实验展示了从不同尺度的噪声开始生成图像的效果。\n")
        f.write("噪声尺度分别为: 0.5, 1.0, 2.0, 4.0\n\n")
        f.write("结论:\n")
        f.write("- 尺度=1.0: 标准高斯噪声，应该产生最佳质量图像\n")
        f.write("- 尺度<1.0: 噪声太小，可能导致模糊或缺乏多样性\n")
        f.write("- 尺度>1.0: 噪声太大，可能导致图像质量下降\n")


    


# 在main函数中添加结果保存功能
def main():
    """运行所有实验"""
    # 创建结果目录
    results_dir = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 实验配置
    experiments = [
        # 基线配置
        Config(timesteps=200, beta_schedule='linear', sampling_method='both', epochs=5),
        
        # 改变噪声调度
        Config(timesteps=200, beta_schedule='cosine', sampling_method='both', epochs=5),
        Config(timesteps=200, beta_schedule='quadratic', sampling_method='both', epochs=5),
        
        # 改变时间步数
        Config(timesteps=100, beta_schedule='linear', sampling_method='both', epochs=5),
        Config(timesteps=500, beta_schedule='linear', sampling_method='both', epochs=5),
        
        # 改变采样方法
        Config(timesteps=200, beta_schedule='linear', sampling_method='ddpm', epochs=5),
        Config(timesteps=200, beta_schedule='linear', sampling_method='ddim', epochs=5),
        
        # 额外实验：余弦调度+更多时间步
        Config(timesteps=500, beta_schedule='cosine', sampling_method='both', epochs=5),
    ]
    
    experiment_names = [
        "baseline",
        "cosine_schedule",
        "quadratic_schedule", 
        "timesteps_100",
        "timesteps_500",
        "ddpm_only",
        "ddim_only",
        "cosine_timesteps_500"
    ]
    
    all_results = []
    
    # 运行所有实验
    for config, name in zip(experiments, experiment_names):
        try:
            # 为每个实验创建子目录
            exp_dir = os.path.join(results_dir, name)
            os.makedirs(exp_dir, exist_ok=True)
            
            # 修改run_experiment函数调用，传入保存目录
            results = run_experiment(config, name, exp_dir)
            all_results.append(results)
            print(f"实验 {name} 完成!")
            print(f"FID分数: {results['fid_scores']}")
            
            # 保存每个实验的详细结果
            exp_results_path = os.path.join(exp_dir, f"{name}_detailed_results.json")
            with open(exp_results_path, 'w') as f:
                # 将配置对象转换为可序列化的字典
                config_dict = {
                    'timesteps': config.timesteps,
                    'beta_schedule': config.beta_schedule,
                    'sampling_method': config.sampling_method,
                    'model_dim': config.model_dim,
                    'img_size': config.img_size,
                    'batch_size': config.batch_size,
                    'lr': config.lr,
                    'epochs': config.epochs
                }
                
                exp_data = {
                    'experiment_name': name,
                    'config': config_dict,
                    'losses': results['losses'],
                    'fid_scores': results['fid_scores'],
                    'final_loss': results['losses'][-1] if results['losses'] else None
                }
                json.dump(exp_data, f, indent=2)
                
        except Exception as e:
            print(f"实验 {name} 失败: {e}")
    
    # 生成总结报告
    print("\n=== 实验总结 ===")
    summary_data = []
    
    for results in all_results:
        name = results['experiment_name']
        fid_scores = results['fid_scores']
        final_loss = results['losses'][-1] if results['losses'] else float('inf')
        
        ddpm_fid = fid_scores.get('ddpm', 'N/A')
        ddim_fid = fid_scores.get('ddim', 'N/A')
        
        summary_data.append({
            'experiment_name': name,
            'ddpm_fid': ddpm_fid if ddpm_fid != 'N/A' else None,
            'ddim_fid': ddim_fid if ddim_fid != 'N/A' else None,
            'final_loss': final_loss,
            'timesteps': results['config'].timesteps,
            'beta_schedule': results['config'].beta_schedule,
            'sampling_method': results['config'].sampling_method
        })
        
        print(f"{name:15} | {ddpm_fid:8.2f} | {ddim_fid:8.2f} | {final_loss:.4f}")
    
    # 保存总结报告为CSV和JSON
    summary_df = pd.DataFrame(summary_data)
    
    # CSV格式
    csv_path = os.path.join(results_dir, "experiment_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    
    # JSON格式
    json_path = os.path.join(results_dir, "experiment_summary.json")
    summary_df.to_json(json_path, indent=2, orient='records')
    
    # 保存详细的文本总结报告
    txt_path = os.path.join(results_dir, "experiment_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=== 扩散模型实验总结报告 ===\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("实验配置:\n")
        f.write("-" * 30 + "\n")
        for exp in summary_data:
            f.write(f"实验名称: {exp['experiment_name']}\n")
            f.write(f"  时间步数: {exp['timesteps']}\n")
            f.write(f"  噪声调度: {exp['beta_schedule']}\n")
            f.write(f"  采样方法: {exp['sampling_method']}\n")
            f.write(f"  最终损失: {exp['final_loss']:.4f}\n")
            if exp['ddpm_fid'] is not None:
                f.write(f"  DDPM FID: {exp['ddpm_fid']:.2f}\n")
            if exp['ddim_fid'] is not None:
                f.write(f"  DDIM FID: {exp['ddim_fid']:.2f}\n")
            f.write("\n")
        
        # 找出最佳实验
        valid_results = [exp for exp in summary_data if exp['ddpm_fid'] is not None or exp['ddim_fid'] is not None]
        if valid_results:
            # 使用DDPM FID作为主要比较指标，如果不可用则用DDIM FID
            def get_best_fid(exp):
                return exp['ddpm_fid'] if exp['ddpm_fid'] is not None else exp['ddim_fid']
            
            best_exp = min(valid_results, key=lambda x: get_best_fid(x))
            f.write("\n最佳实验:\n")
            f.write("-" * 20 + "\n")
            f.write(f"实验名称: {best_exp['experiment_name']}\n")
            f.write(f"时间步数: {best_exp['timesteps']}\n")
            f.write(f"噪声调度: {best_exp['beta_schedule']}\n")
            f.write(f"采样方法: {best_exp['sampling_method']}\n")
            if best_exp['ddpm_fid'] is not None:
                f.write(f"DDPM FID: {best_exp['ddpm_fid']:.2f}\n")
            if best_exp['ddim_fid'] is not None:
                f.write(f"DDIM FID: {best_exp['ddim_fid']:.2f}\n")
    
    print(f"\n所有结果已保存到目录: {results_dir}")
    print(f"总结报告: {txt_path}")
    print(f"CSV格式: {csv_path}")
    print(f"JSON格式: {json_path}")
    
    # 找到最佳模型进行潜在质量实验
    valid_results = [r for r in all_results if r['fid_scores']]
    if valid_results:
        best_result = min(valid_results, key=lambda x: min(x['fid_scores'].values()))
        best_config = best_result['config']
        
        print(f"\n最佳模型: {best_result['experiment_name']}")
        print(f"最佳FID: {best_result['fid_scores']}")
        
        # 重新训练最佳模型进行潜在质量实验
        print("\n重新训练最佳模型进行潜在质量实验...")
        best_model, best_diffusion, _ = train_model(best_config)
        
        # 保存潜在质量实验结果
        latent_dir = os.path.join(results_dir, "latent_quality_experiment")
        os.makedirs(latent_dir, exist_ok=True)
        
        demonstrate_latent_quality(best_model, best_diffusion, best_config, latent_dir)
        
        print(f"潜在质量实验结果保存到: {latent_dir}")



if __name__ == "__main__":
    # 确保必要的导入
    from functools import partial
    from scipy.linalg import sqrtm
    
    main()
