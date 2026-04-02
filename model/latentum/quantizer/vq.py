import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit import ViT


def get_vq_quantizer(config):
    if config.type == "MLP":
        return VQ_MLP(config)
    elif config.type == "ViT":
        return VQ_ViT(config)
    else:
        raise NotImplementedError


class VectorQuantizer(nn.Module):
    """
    传统的Vector Quantizer模块
    使用codebook进行离散量化，通过STE进行梯度传播
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, ema_decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost if commitment_cost is not None else 0.25
        self.ema_decay = ema_decay

        # codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA 统计：追踪每个 code 的使用频率
        self.register_buffer("ema_usage", torch.zeros(num_embeddings))
        # 这里的初始值设为均匀分布，避免刚开始就误判为死码
        self.ema_usage.fill_(1.0 / num_embeddings)

    def forward(self, z):
        """
        :param z: (B, L, D), 连续的latent表示
        :return: z_q: 量化后的表示, indices: codebook索引, vq_loss: VQ损失
        """
        B, L, D = z.shape
        z_flat = z.reshape(-1, D)  # (B*L, D)

        # 计算与codebook的距离: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        # (B*L, num_embeddings)
        d2 = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
             torch.sum(self.codebook.weight ** 2, dim=1) - \
             2 * torch.matmul(z_flat, self.codebook.weight.t())

        # 找到最近的codebook向量
        indices = torch.argmin(d2, dim=1)  # (B*L,)
        z_q_flat = self.codebook(indices)  # (B*L, D)
        z_q = z_q_flat.reshape(B, L, D)  # (B, L, D)

        # VQ损失: codebook loss + commitment loss
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        
        # 3. Entropy Loss (新增): 鼓励 code 使用分布均匀，最大化熵 = 最小化 -entropy
        # 计算当前 batch 的 code 使用概率
        # 使用 softmax(-distance) 作为软分配概率，比硬分配 indices 更平滑，梯度更好
        # 这里的 temperature 可以调节，通常取 1.0 或更小
        probs = F.softmax(-d2, dim=1)  # (B*L, num_embeddings)
        avg_probs = probs.mean(dim=0)  # (num_embeddings,)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        # 这是一个正则项，通常系数不用太大，比如 0.1 左右
        # 我们这里直接加到 vq_loss 里，系数设为 0.1 (经验值)
        entropy_loss = -entropy

        # 分开记录各损失分量
        codebook_commitment_loss = codebook_loss + self.commitment_cost * commitment_loss
        scaled_entropy_loss = 0.1 * entropy_loss
        vq_loss = codebook_commitment_loss + scaled_entropy_loss
        
        vq_loss_dict = {
            "total": vq_loss,
            "codebook_commitment": codebook_commitment_loss,
            "entropy": scaled_entropy_loss,
        }

        # Straight-Through Estimator: 前向用z_q，反向用z的梯度
        z_q = z + (z_q - z).detach()

        # 更新 EMA 使用统计（仅在训练时）
        if self.training:
            with torch.no_grad():
                # 1. 更新 EMA
                encodings = F.one_hot(indices, self.num_embeddings).float()
                batch_usage = encodings.mean(dim=0)  # 当前 batch 的使用频率
                self.ema_usage = self.ema_decay * self.ema_usage + (1 - self.ema_decay) * batch_usage

                # 2. 死码重置 (Codebook Restart)
                # 阈值：如果使用频率低于均匀分布的 3% (经验值)，认为它是死码
                usage_threshold = 0.03 / self.num_embeddings
                
                # 找出死码
                dead_mask = self.ema_usage < usage_threshold
                if dead_mask.any():
                    dead_indices = torch.nonzero(dead_mask).squeeze(-1)
                    
                    # 限制每次重置的数量，避免震荡 (例如每次最多重置 10% 的死码 或 固定数量)
                    # 这里我们限制每个 step 最多重置 64 个，慢慢救活
                    num_reset = min(64, dead_indices.numel())
                    
                    # 随机选择要重置的死码
                    perm_dead = torch.randperm(dead_indices.numel(), device=dead_indices.device)[:num_reset]
                    reset_indices = dead_indices[perm_dead]
                    
                    # 从当前 batch 的输入 z 中随机采样作为新的 codebook 向量
                    # 这样可以保证新的 code 位于真实数据分布上
                    if B * L >= num_reset:
                        perm_z = torch.randperm(B * L, device=z.device)[:num_reset]
                        new_codes = z_flat[perm_z].detach()
                    else:
                        # 如果 batch size 不够大，允许重复采样
                        rand_idx = torch.randint(0, B * L, (num_reset,), device=z.device)
                        new_codes = z_flat[rand_idx].detach()
                    
                    # 执行重置
                    self.codebook.weight.data[reset_indices] = new_codes
                    
                    # 重置这些 code 的 EMA 统计，给它们“重新做人”的机会
                    self.ema_usage[reset_indices] = 1.0 / self.num_embeddings

        indices = indices.reshape(B, L)  # (B, L)

        return z_q, indices, vq_loss_dict
    
    def get_codebook_usage(self):
        """
        获取 codebook 的利用率指标（基于 EMA 统计）
        
        :return: dict 包含:
            - perplexity: 困惑度，越高表示利用率越高，最大值 = num_embeddings
            - utilization: 使用率，有多少比例的 code 被"有效"使用 (使用频率 > 阈值)
            - active_codes: 活跃的 code 数量
        """
        # Perplexity: exp(-sum(p * log(p)))
        probs = self.ema_usage + 1e-10  # 避免 log(0)
        probs = probs / probs.sum()  # 确保归一化
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs)))
        
        # 活跃 code：使用频率高于均匀分布的 1/10
        threshold = 0.1 / self.num_embeddings
        active_codes = (self.ema_usage > threshold).sum().item()
        utilization = active_codes / self.num_embeddings
        
        return {
            "perplexity": perplexity.item(),
            "utilization": utilization,
            "active_codes": active_codes,
        }


class VQ_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.down_proj = nn.Sequential(
            nn.Linear(config.input_feature_dim, 4 * config.input_feature_dim),
            nn.GELU(),
            nn.Linear(4 * config.input_feature_dim, config.embedding_dim),
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            commitment_cost=getattr(config, "commitment_cost", None) or 0.25,
        )

        self.up_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.llm_hidden_size, config.llm_hidden_size),
        )

    def forward(self, x):
        """
        :param x: (B, 256, 4096), the input clip features
        :return: x_vq: 重建的特征, indices: codebook索引, vq_loss_dict: VQ损失字典
        """
        z = self.down_proj(x)  # (B, 256, output_dim)
        z_q, indices, vq_loss_dict = self.quantizer(z)  # (B, 256, output_dim)
        x_vq = self.up_proj(z_q)  # (B, 256, llm_hidden_size)

        return x_vq, indices, vq_loss_dict

    @torch.no_grad()
    def get_ids(self, x):
        z = self.down_proj(x)
        _, indices, _ = self.quantizer(z)
        return indices


class VQ_ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.down_proj = ViT(config)  # 4096 -> output_dim

        self.quantizer = VectorQuantizer(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.output_dim,
            commitment_cost=getattr(config, "commitment_cost", None) or 0.25,
        )

        self.up_proj = nn.Sequential(
            nn.Linear(config.output_dim, 4 * config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.llm_hidden_size, config.llm_hidden_size),
        )

    def forward(self, x):
        """
        :param x: (B, 256, 4096), the input clip features
        :return: x_vq: 重建的特征, indices: codebook索引, vq_loss_dict: VQ损失字典
        """
        z = self.down_proj(x)  # (B, 256, output_dim)
        z_q, indices, vq_loss_dict = self.quantizer(z)  # (B, 256, output_dim)
        x_vq = self.up_proj(z_q)  # (B, 256, llm_hidden_size)

        return x_vq, indices, vq_loss_dict

    @torch.no_grad()
    def get_ids(self, x):
        z = self.down_proj(x)
        _, indices, _ = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode_from_ids(self, indices):
        """
        从codebook索引解码
        :param indices: (B, L), codebook索引
        :return: x_vq: (B, L, llm_hidden_size)
        """
        z_q = self.quantizer.codebook(indices)  # (B, L, output_dim)
        x_vq = self.up_proj(z_q)  # (B, L, llm_hidden_size)
        return x_vq


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("config/vq_distill/distill_vq.yaml")
    
    # # 需要在config中添加 num_embeddings 参数
    # config.model.quantizer. = 65536

    model = get_vq_quantizer(config.model.quantizer)
    x = torch.randn(2, 256, 4096)
    x_vq, indices, vq_loss_dict = model(x)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params/1e6:.2f}M")
    print(f"x_vq shape: {x_vq.shape}")
    print(f"indices shape: {indices.shape}")
    print(f"vq_loss (total): {vq_loss_dict['total'].item():.4f}")
    print(f"vq_loss (codebook_commitment): {vq_loss_dict['codebook_commitment'].item():.4f}")
    print(f"vq_loss (entropy): {vq_loss_dict['entropy'].item():.4f}")
