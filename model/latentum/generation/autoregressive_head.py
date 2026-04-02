import torch
import torch.nn as nn
from einops import rearrange
from .autoregressive_head_components import DecoderLayer, RMSNorm, precompute_freqs_cis_1d
from ..sample import sample

class AutoregressiveHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList()
        for _ in range(self.config.num_codebooks):
            embedding = nn.Embedding(self.config.num_embeddings, self.config.hidden_size)
            self.embeddings.append(embedding)
        
        self.layers = nn.ModuleList()
        for _ in range(self.config.num_layers):
            layer = DecoderLayer(self.config)
            self.layers.append(layer)
        
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)
        self.head = nn.Linear(self.config.hidden_size, self.config.num_embeddings)
        
        # 1D RoPE 位置编码缓存
        self.precompute_pos = {}
    
    def get_freqs_cis(self, seq_len: int, device):
        """获取或计算指定长度的位置编码"""
        if seq_len not in self.precompute_pos:
            head_dim = self.config.hidden_size // self.config.num_heads
            freqs_cis = precompute_freqs_cis_1d(head_dim, seq_len)
            self.precompute_pos[seq_len] = freqs_cis
        return self.precompute_pos[seq_len].to(device)

    def forward(self, x):
        """
        x: (B, N, D) input embeddings
        returns: (B, N, V) logits
        """
        B, N, D = x.shape
        pos = self.get_freqs_cis(N, x.device)
        
        for layer in self.layers:
            x = layer(x, pos)
        
        x = self.norm(x)
        logits = self.head(x)

        return logits

    def _code_to_embeddings(self, code):
        """
        code: (B, L, K)
        return: (BxL, K, D)
        """
        B, L, K = code.shape
        code = rearrange(code, "B L K -> (B L) K")
        index_embeddings = []
        for i in range(K):
            index_embed = self.embeddings[i](code[:, i])
            index_embeddings.append(index_embed)
        index_embeddings = torch.stack(index_embeddings, dim=1)

        return index_embeddings

    def generate_from_base_token(self, base_token, cfg_scale, sampling_kwargs):
        """
        base_token: (B, 1, D) if cfg_scale <= 1, else (2*B, 1, D) where first B is cond, last B is uncond
        returns: (B, K) generated code indices
        """
        generated_code = []
        if cfg_scale > 1:
            B = base_token.shape[0] // 2
            # 分离条件和无条件状态
            curr_state_cond = base_token[:B]  # (B, 1, D)
            curr_state_uncond = base_token[B:]  # (B, 1, D)
            
            for i in range(self.config.num_codebooks):
                # 分别计算条件和无条件的 logits
                logits_cond = self.forward(curr_state_cond)[:, -1, :]  # (B, V)
                logits_uncond = self.forward(curr_state_uncond)[:, -1, :]  # (B, V)
                
                # 应用 CFG 公式: logits = uncond + cfg_scale * (cond - uncond)
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)  # (B, V)
                
                # 从合并后的 logits 采样
                next_token, _ = sample(logits, **sampling_kwargs)  # next_token: (B, 1)
                generated_code.append(next_token)
                
                # 条件和无条件分支使用相同的采样 token 更新状态
                next_embeddings = self.embeddings[i](next_token)  # (B, 1, D)
                curr_state_cond = torch.cat([curr_state_cond, next_embeddings], dim=1)
                curr_state_uncond = torch.cat([curr_state_uncond, next_embeddings], dim=1)
            
            generated_code = torch.stack(generated_code, dim=1).squeeze(-1)  # (B, K)
            return generated_code
        else:
            curr_state = base_token  # (B, 1, D)
            for i in range(self.config.num_codebooks):
                logits = self.forward(curr_state)[:, -1, :]  # (B, V)
                next_token, _ = sample(logits, **sampling_kwargs)  # next_token: (B, 1)
                generated_code.append(next_token)
                next_embeddings = self.embeddings[i](next_token)  # (B, 1, D)
                curr_state = torch.cat([curr_state, next_embeddings], dim=1)  # (B, i+2, D)
            generated_code = torch.stack(generated_code, dim=1).squeeze(-1)  # (B, K)
            return generated_code

    def generate_from_base_token_with_logps(self, base_token, cfg_scale, sampling_kwargs):
        """
        从 base token 生成 K 个码本索引，同时记录 log prob

        Args:
            base_token: (B, 1, D) 输入 hidden state
            cfg_scale: float CFG 缩放因子
            sampling_kwargs: dict 采样参数

        Returns:
            generated_code: (B, K) 生成的 code indices
            code_logps: (B, K) 对应的 log probs
        """
        import torch.nn.functional as F

        generated_code = []
        code_logps = []
        curr_state = base_token  # (B, 1, D)

        for i in range(self.config.num_codebooks):
            logits = self.forward(curr_state)[:, -1, :]  # (B, V)

            # 计算 log prob
            log_probs = F.log_softmax(logits, dim=-1)

            # 采样
            next_token, _ = sample(logits, **sampling_kwargs)  # (B, 1)

            # 记录采样 token 的 log prob
            token_logp = log_probs.gather(1, next_token)  # (B, 1)

            generated_code.append(next_token)
            code_logps.append(token_logp)

            # 更新状态
            next_embeddings = self.embeddings[i](next_token)  # (B, 1, D)
            curr_state = torch.cat([curr_state, next_embeddings], dim=1)

        generated_code = torch.stack(generated_code, dim=1).squeeze(-1)  # (B, K)
        code_logps = torch.stack(code_logps, dim=1).squeeze(-1)  # (B, K)

        return generated_code, code_logps

    def compute_log_prob(self, base_token, target_codes):
        """
        计算给定 codes 的 log prob（用于 PPO 更新时重新计算当前策略的 log prob）

        Args:
            base_token: (B, 1, D) 输入 hidden state
            target_codes: (B, K) 目标 code indices

        Returns:
            logps: (B, K) log probs
        """
        import torch.nn.functional as F

        logps = []
        curr_state = base_token  # (B, 1, D)

        for i in range(self.config.num_codebooks):
            # 前向获取 logits
            logits = self.forward(curr_state)[:, -1, :]  # (B, V)

            # 计算 log prob
            log_probs = F.log_softmax(logits, dim=-1)

            # 获取目标 token 的 log prob
            target = target_codes[:, i:i+1]  # (B, 1)
            logp = log_probs.gather(1, target)  # (B, 1)
            logps.append(logp)

            # 使用目标 token 更新状态（teacher forcing）
            next_embeddings = self.embeddings[i](target)  # (B, 1, D)
            curr_state = torch.cat([curr_state, next_embeddings], dim=1)

        return torch.cat(logps, dim=1)  # (B, K)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/mcq_gen/dev_autoregressive_head.yaml")

    model = AutoregressiveHead(config.model.head)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model = model.to(device, dtype)

    # x = torch.randn(2, 8, 2560).to(device, dtype)
    # logits = model(x)
    # print(logits.shape)

    code = torch.randint(0, 2048, (2, 256, 8)).to(device, torch.int32)
    index_embeddings = model._code_to_embeddings(code)
    print(index_embeddings.shape)
