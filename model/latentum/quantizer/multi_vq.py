import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import get_norm_layer

from .vq import VectorQuantizer


class GeGluMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        act_layer = None,
        drop = 0.0,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer('layernorm'), eps=1e-6)
        self.norm = norm_layer(in_features)
        self.act = nn.GELU(approximate='tanh')
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x

class PlainAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # assert in_dim // num_heads == out_dim
            self.head_dim = in_dim // num_heads
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        else:
            # assert out_dim // num_heads == in_dim
            self.head_dim = out_dim // num_heads
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            self.register_buffer("zero_k_bias", torch.zeros(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)

        if self.in_dim > self.out_dim:
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        return x

class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = norm_layer(in_dim)
        self.attn = PlainAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = norm_layer(in_dim)

        self.norm2 = norm_layer(out_dim)
        hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim
        )

    def forward(self, x):
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiVectorQuantizer(nn.Module):
    """
    Multi-Head Vector Quantizer (Product Quantization)
    
    将输入特征维度切分为 num_codebooks 份，每份独立使用一个 VectorQuantizer 进行量化。
    最后将量化后的特征拼接回原始维度。
    
    优点：
    1. 组合爆炸：N个大小为K的codebook可以表达 K^N 种状态。
    2. 计算效率：每个子空间的维度更小，计算距离更快。
    """
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim, 
        num_codebooks, 
        commitment_cost=0.25, 
        ema_decay=0.99,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embedding_dim = embedding_dim
        
        assert embedding_dim % num_codebooks == 0, \
            f"embedding_dim {embedding_dim} must be divisible by num_codebooks {num_codebooks}"
        
        self.dim_per_codebook = embedding_dim // num_codebooks
        
        # 创建多个独立的 VectorQuantizer
        # 使用 nn.ModuleList 注册子模块
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                num_embeddings           = num_embeddings,
                embedding_dim            = self.dim_per_codebook,
                commitment_cost          = commitment_cost,
                ema_decay                = ema_decay,
            )
            for _ in range(num_codebooks)
        ])

    def forward(self, z):
        """
        :param z: (B, L, D) 输入特征
        :return: 
            z_q: (B, L, D) 量化后的特征 (拼接后)
            indices: (B, L, num_codebooks) 每个head的codebook索引
            vq_loss_dict: 损失字典 (所有head的平均)
        """
        B, L, D = z.shape
        assert D == self.embedding_dim
        
        # 将特征切分为 (B, L, num_codebooks, dim_per_codebook)
        # 但为了方便分别送入 quantizers，我们先在最后一维 split
        z_splits = torch.split(z, self.dim_per_codebook, dim=-1)
        
        z_q_list = []
        indices_list = []
        total_vq_loss = 0.0
        total_codebook_commitment = 0.0
        total_entropy = 0.0
        
        # 对每个切片分别进行量化
        for i, quantizer in enumerate(self.quantizers):
            # z_chunk: (B, L, dim_per_codebook)
            z_chunk = z_splits[i]
            
            # z_q_chunk: (B, L, dim_per_codebook)
            # indices_chunk: (B, L)
            z_q_chunk, indices_chunk, loss_dict_chunk = quantizer(z_chunk)
            
            z_q_list.append(z_q_chunk)
            indices_list.append(indices_chunk)
            total_vq_loss += loss_dict_chunk["total"]
            total_codebook_commitment += loss_dict_chunk["codebook_commitment"]
            total_entropy += loss_dict_chunk["entropy"]
            
        # 拼接回原始维度
        z_q = torch.cat(z_q_list, dim=-1) # (B, L, D)
        
        # 堆叠索引 (B, L, num_codebooks)
        indices = torch.stack(indices_list, dim=-1)
        
        # Loss 取平均
        vq_loss_dict = {
            "total": total_vq_loss / self.num_codebooks,
            "codebook_commitment": total_codebook_commitment / self.num_codebooks,
            "entropy": total_entropy / self.num_codebooks,
        }
        
        return z_q, indices, vq_loss_dict

    def get_codebook_usage(self):
        """
        获取所有 codebook 的平均利用率指标
        """
        total_perplexity = 0.0
        total_utilization = 0.0
        total_active = 0.0
        
        stats_list = []
        
        for i, quantizer in enumerate(self.quantizers):
            stats = quantizer.get_codebook_usage()
            stats_list.append(stats)
            total_perplexity += stats["perplexity"]
            total_utilization += stats["utilization"]
            total_active += stats["active_codes"]
            
        return {
            "perplexity": total_perplexity / self.num_codebooks,
            "utilization": total_utilization / self.num_codebooks,
            "active_codes": total_active / self.num_codebooks,
            "per_codebook_stats": stats_list
        }

class VQ_MLP_MCQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 确保配置里有 num_codebooks
        num_codebooks = getattr(config, "num_codebooks", 4) # 默认4个头

        self.down_proj = nn.Sequential(
            nn.Linear(config.input_feature_dim, 4 * config.input_feature_dim),
            nn.GELU(),
            nn.Linear(4 * config.input_feature_dim, config.embedding_dim),
        )

        self.quantizer = MultiVectorQuantizer(
            num_embeddings  = config.num_embeddings,
            embedding_dim   = config.embedding_dim,
            num_codebooks   = num_codebooks,
            commitment_cost = getattr(config, "commitment_cost", None) or 0.25,
        )

        self.up_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.llm_hidden_size, config.llm_hidden_size),
        )

    def forward(self, x):
        z = self.down_proj(x)
        z_q, indices, vq_loss_dict = self.quantizer(z)
        x_vq = self.up_proj(z_q)
        return x_vq, indices, vq_loss_dict
    
    def get_zq_indices(self, x):
        z = self.down_proj(x)
        z_q, indices, _ = self.quantizer(z)
        return z_q, indices
    
    def to_abs_code(self, indices):
        """
        输入的indices的index都是[0, 2048], 目标是把第i个codebook的index范围改成[i*2048, (i+1)*2048)], 并展平
        indices: (B, L, num_codebooks)
        return: (B, Lxnum_codebooks)
        """
        B, L, num_codebooks = indices.shape
        num_embeddings = self.config.num_embeddings
        
        # 为每个codebook的索引添加偏移量
        # 第i个codebook的索引范围从[0, num_embeddings)变成[i*num_embeddings, (i+1)*num_embeddings)
        offsets = torch.arange(num_codebooks, device=indices.device) * num_embeddings
        # offsets: (num_codebooks,) -> (1, 1, num_codebooks)
        offsets = offsets.view(1, 1, num_codebooks)
        
        # 添加偏移量
        abs_indices = indices + offsets  # (B, L, num_codebooks)
        
        # 展平最后两个维度: (B, L, num_codebooks) -> (B, L*num_codebooks)
        abs_indices = abs_indices.reshape(B, L * num_codebooks)
        
        return abs_indices

    def to_rel_code(self, indices):
        """
        to_abs_code 的逆变换：将绝对索引转换回相对索引
        输入的indices是绝对索引，范围是[0, num_codebooks*num_embeddings)
        目标是把第i个codebook的索引范围从[i*num_embeddings, (i+1)*num_embeddings)改回[0, num_embeddings)
        indices: (B, L*num_codebooks) 绝对索引
        return: (B, L, num_codebooks) 相对索引
        """
        B, flat_len = indices.shape
        num_codebooks = self.quantizer.num_codebooks
        num_embeddings = self.config.num_embeddings
        
        # 从展平的形状恢复: (B, L*num_codebooks) -> (B, L, num_codebooks)
        L = flat_len // num_codebooks
        assert flat_len % num_codebooks == 0, f"flat_len {flat_len} must be divisible by num_codebooks {num_codebooks}"
        rel_indices = indices.reshape(B, L, num_codebooks)
        
        # 为每个codebook计算偏移量并减去
        # 第i个codebook的索引范围从[i*num_embeddings, (i+1)*num_embeddings)变回[0, num_embeddings)
        offsets = torch.arange(num_codebooks, device=indices.device) * num_embeddings
        # offsets: (num_codebooks,) -> (1, 1, num_codebooks)
        offsets = offsets.view(1, 1, num_codebooks)
        
        # 减去偏移量，将绝对索引转换回相对索引
        rel_indices = rel_indices - offsets  # (B, L, num_codebooks)
        
        return rel_indices

    @torch.no_grad()
    def indices_to_feature(self, indices):
        """
        从 VQ 索引反向查找得到量化后的特征
        
        :param indices: (B, L, num_codebooks) 每个位置在各个 codebook 中的索引
        :return: (B, L, llm_hidden_size) 经过 up_proj 后的特征
        """
        B, L, num_codebooks = indices.shape

        # 从每个 codebook 查找对应的 embedding，然后拼接
        z_q_list = []
        for i, quantizer in enumerate(self.quantizer.quantizers):
            # indices_chunk: (B, L)
            indices_chunk = indices[..., i]
            # z_q_chunk: (B, L, dim_per_codebook)
            z_q_chunk = quantizer.codebook(indices_chunk)
            z_q_list.append(z_q_chunk)

        # 拼接得到 (B, L, embedding_dim)
        z_q = torch.cat(z_q_list, dim=-1)

        # 通过 up_proj 得到最终特征
        x_vq = self.up_proj(z_q)
        
        return z_q, x_vq

class VQ_Attn_MCQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pre_quant_proj = AttnProjection(config.input_feature_dim, config.embedding_dim, config.input_feature_dim // config.embedding_dim)

        self.quantizer = MultiVectorQuantizer(
            num_embeddings  = config.num_embeddings,
            embedding_dim   = config.embedding_dim,
            num_codebooks   = config.num_codebooks,
            commitment_cost = getattr(config, "commitment_cost", None) or 0.25,
        )

        self.post_quant_proj = AttnProjection(config.embedding_dim, config.input_feature_dim, config.input_feature_dim // config.embedding_dim)

        self.up_proj = nn.Sequential(
            nn.Linear(config.input_feature_dim, 4 * config.llm_hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.llm_hidden_size, config.llm_hidden_size),
        )

    def forward(self, x):
        z = self.pre_quant_proj(x)
        z_q, indices, vq_loss_dict = self.quantizer(z)
        z_q = self.post_quant_proj(z_q)
        x_vq = self.up_proj(z_q)

        return x_vq, indices, vq_loss_dict

    @torch.no_grad()
    def indices_to_feature(self, indices):
        """
        从 VQ 索引反向查找得到量化后的特征
        
        :param indices: (B, L, num_codebooks) 每个位置在各个 codebook 中的索引
        :return: (B, L, llm_hidden_size) 经过 post_quant_proj 和 up_proj 后的特征
        """
        B, L, num_codebooks = indices.shape

        # 从每个 codebook 查找对应的 embedding，然后拼接
        z_q_list = []
        for i, quantizer in enumerate(self.quantizer.quantizers):
            indices_chunk = indices[..., i]
            z_q_chunk = quantizer.codebook(indices_chunk)
            z_q_list.append(z_q_chunk)

        # 拼接得到 (B, L, embedding_dim)
        z_q = torch.cat(z_q_list, dim=-1)

        # 通过 post_quant_proj 和 up_proj 得到最终特征
        z_q = self.post_quant_proj(z_q)
        x_vq = self.up_proj(z_q)

        return z_q, x_vq

def get_multi_vq_quantizer(config):
    if config.type == "MLP":
        return VQ_MLP_MCQ(config)
    elif config.type == "Attn":
        return VQ_Attn_MCQ(config)
    else:
        raise ValueError(f"Unsupported quantizer type: {config.type}")

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/vq_distill/distill_multivq_2.yaml")
    quantizer = get_multi_vq_quantizer(config.model.quantizer)
    num_paras = sum(p.numel() for p in quantizer.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_paras/1e6:.2f}M")
    x = torch.randn(2, 256, 4096)
    x_vq, indices, vq_loss_dict = quantizer(x)
    print(x_vq.shape, indices.shape, vq_loss_dict)

    z_q, x_vq_ = quantizer.indices_to_feature(indices)
    print(z_q.shape, x_vq.shape)
    print((x_vq - x_vq_).abs().max())
