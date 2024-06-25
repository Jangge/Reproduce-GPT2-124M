from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

# -----------------------------------------------------------------
"""超参数"""
@dataclass
class GPTConfig:
    block_size: int = 1024   # 上下文token，一次最长输入256个词
    vocab_size: int = 50257    # 字典大小
    n_layer: int = 12        # 注意力层数
    n_head: int = 12         # 注意力的头
    n_embd: int = 768      # 词嵌入的维度大小 




# -----------------------------------------------------------------
"""主题思想：想搭建大的框架，然后逐渐补充内容，GPT2--BLOCK---"""

# 第四步搭建多头自注意力
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0                          # 断言句，上来就要判断词嵌入维度能否被头数整除

        # 通过词向量映射key, query, value，因为是三个，所以映射到三倍大小
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)           
        
        # 输出投影，映射为原始嵌入维度
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # 加载heads和embed
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # 注意力遮盖，确保每个位置的注意力只能看到前面的词，也就是掩蔽自注意力中的掩蔽
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):

        # B：批量大小， T：序列长度， C：嵌入维度
        B, T, C = x.size()

        # 将q, k ,v从词向量的映射中拆分出来
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 调整qkv的形状
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 此时的hs是原来词嵌入大小整除head数
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 

        # 注意力计算
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T]== 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        
        # 这这里使用了FlashAttention计算方式，是一种加速训练计算技巧！
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 将注意力输出形状转换为原来的B, T, C
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 对注意力结果进行线性映射
        y = self.c_proj(y)

        return y


# 第三步搭建MLP
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)   # 两层线性层，这是第一层
        self.gelu = nn.GELU(approximate='tanh')                     # GLUE激活函数，不同于RELU更加平滑，在负值也会提供一点梯度，所以平滑
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # 第二层在转为原来的维度

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# 第二步搭建Block

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)    # 层归一化 
        self.attn = CausalSelfAttention(config)     # 需要构建的多头自注意力，一种归约
        self.ln_2 = nn.LayerNorm(config.n_embd)    # 层归一化
        self.mlp = MLP(config)                      # 需要构建的MLLP全连接层

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))             # 注意这里+表示残差连接
        x = x + self.mlp(self.ln_2(x))              # 同样的数据流,MLP只起映射作用
        return x


# 第一步先搭建整体框架
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        """模型框架"""
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # 对token进行词嵌入
            wpe = nn.Embedding(config.block_size, config.n_embd), # 位置编码的词嵌入
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)), # 通过迭代的形式构建12个注意力层，其中的Block需要定义
            ln_f = nn.LayerNorm(config.n_embd), # 层归一化，这是相较于原transformer解码器的改变地方
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # 对应输出的线性层，输出一个字典大小的数值表示概率

        # 权重共享？
        self.transformer.wte.weight = self.lm_head.weight

        # 参数初始化
        self.apply(self._init_weights)

    # 去做那种初始化函数
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    
    # 前向传播过程
    def forward(self, idx, targets=None):
        # idx is of shape(B, T)
        B, T = idx.size()

        # 断言序列长度一定要小于模型输入长度
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # tiken的前向传播过程
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)             # 位置编码的embedding
        tok_emb = self.transformer.wte(idx)             # token的词嵌入
        x = tok_emb + pos_emb                           # 词嵌入和位置编码结合

        # 注意力层的前向传播过程
        for block in self.transformer.h:
            x = block(x)

        # 层归一化和最后的分类层传播过程
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                        # (B, T, vocab_size)表示每个词的概率

        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss                             # logits:字典对应每个词的概率， loss:和下一个预测词的的交叉损失



    """ 定义一个类方法用于直接从huggface加载预训练好的GPT2模型权重 """
    @classmethod
    def from_pretrained(cls, model_type):

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 定义不同模型的大小的参数
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600),# 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # 先根据参数创建一个model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # 加载模型参数
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 这段没搞懂
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys:{len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):

                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    # 通过将模型参数分组并应用不同的权重衰减策略
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # 返回所有需要更新梯度的参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 创建参数分组，所有2维及以上的参数都需要权重衰减（如权重矩阵），其它不用（如偏执和层归一化）
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay':weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 计算每组参数的数量和总数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        print(f"num decayed parameter tensor : {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensor: {len(nodecay_params)}, with {num_nodecay_params:,} paramters")
        # 检查并适用fused版本的AdamW优化器
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        # if master_process:
        print(f"using fused AdamW:{use_fused}")
        # 创建并返回优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer