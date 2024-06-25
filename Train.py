"""训练代码"""

from GPT2 import GPT, GPTConfig
from Dataloader import DataLoaderLite
from utils.data_hellaswag import render_example, iterate_examples, get_most_likely_row

import torch
import time
import math
import os
import tiktoken
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F



# -------------------------------------------------------------------------------------------------------------
"""0、预先加载的一些工具"""
# 设定训练设备
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# 设定随机数种子
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
# 设定分词的形式
enc = tiktoken.get_encoding("gpt2")


# -------------------------------------------------------------------------------------------------------------
""" 1、加载训练数据集"""
total_batch_size = 524288                       # 2**19,约0.5M的token数量
B = 16                                          # 小batch的大小，3090只能支持16，教程中可以是64
T = 1024                                        # 使用21GB显存！！显卡只能使用这么大的了
assert total_batch_size % (B * T) == 0          # 确保大批量是小批量的整数倍！
grad_accum_steps = total_batch_size // (B * T)  # 这个是统计需要进行累积小批量的轮数
print(f"batch size: {total_batch_size} little batch:{B * T}, Need_steps:{grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T, split='train')  # 训练集
val_loader = DataLoaderLite(B=B, T=T, split='val')      # 验证集
print("Finish train and val dataloader!")
# 加速训练-1：使用FP32浮点数，而不是默认的FP64浮点数
torch.set_float32_matmul_precision('high')



# -------------------------------------------------------------------------------------------------------------
""" 2、加载模型"""
model = GPT(GPTConfig(vocab_size=50304))        # 使用随机权重开始训练
# model = GPT.from_pretrained('gpt2')           # 使用预训练的权重
model.to(device)
# 加速训练-2：将模型进行编译和转化，提高模型训练和推理速度
# model = torch.compile(model)                  # python = 3.12不支持！！需要3.11，而且模型中出现中文注释也不行
print("Success load the model!")


# -------------------------------------------------------------------------------------------------------------
"""3、学习率设定"""
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 16

def get_lr(it):
    # 线性热身阶段
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 学习率衰减结束后，限制最小学习率为min_lr
    if it > max_steps:
        return min_lr
    # 余弦学习率衰减阶段
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# -------------------------------------------------------------------------------------------------------------
""" 3、定义日志文件保存"""
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # 打开并关闭，用于清空内容，“w”用于写入模式，会清空原有内容
    pass


# -------------------------------------------------------------------------------------------------------------
""" 4、定义参数优化器"""
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)


# -------------------------------------------------------------------------------------------------------------
""" 5、一个训练循环(注意此处只是使用了了20个batch,不是epoch)"""
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    #------------------------------------------------------
    # A、在验证集上验证模型准确度，并保存模型训练检查点权重
    if step % 2 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 2     # 选取2小小批量进行验证
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f"step:{step} : validation loss {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        # 保存模型检查点信息
        if step > 0 and (step % 4 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_in_step{step}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config':model.config,
                'step':step,
                'val_loss':val_loss_accum.item()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Success save the model with val_loss:{val_loss_accum.item():.2f} in step:{step}")
        

    #------------------------------------------------------
    # B、在hellaseag数据集上验证模型准确度
    if step % 2 == 0 or last_step:
        num_correct_norm = 0
        num_total = 0 
        for i, example in enumerate(iterate_examples("val")):
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # 获取输出值
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        acc_norm = num_correct_norm / num_total
        print(f"HellaSwag accuracy:{num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f: # 'a'是追加模式
            f.write(f"{step} hella {acc_norm:.4f}\n")


    #------------------------------------------------------
    # C、观察模型的输出
    if step > 0 and step % 2 == 0 or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a labguage model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        while xgen.size(1) < max_length:
            # 前向传播获取预测结果
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # 将预测结果放在最后
                logits = logits[:, -1, :] # (B, vocab_size)
                # 将结果softmax化
                probs = F.softmax(logits, dim=-1)
                # 做topk的选择
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # 随机从topk中选择一个输出,ix是索引
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # 从topk_indices 中获取对应于采样索引 ix 的词汇索引。xcol 是新生成的词汇索引，形状为 (B, 1)
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # 添加到输出序列中
                xgen = torch.cat((xgen, xcol), dim=1)

        # 打印输出
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample{i}:{decoded}")


    #------------------------------------------------------
    # D、正常的训练步骤
    optimizer.zero_grad()
    loss_accum = 0.0

    # 先进行小批量累积梯度
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # 加速训练-3：在前向传播和损失计算中使用BF16浮点数，只有之两个参数是的，也就是混合精度的来源
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # 因为累加的loss没有平均，所有需要加上
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    # 梯度裁剪
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 学习率根据设定变化
    lr = get_lr(step)
    for param_group in optimizer.param_groups: # 遍历参数组，将所有参数组的学习率设置为lr
        param_group['lr'] = lr

    # 更新参数
    optimizer.step()

    torch.cuda.synchronize() # 等待GPU完成计算工作

    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_pre_sec = tokens_processed / dt
    print(f"step: {step}, | loss:{loss_accum.item():.2f}, | lr:{lr:.4e} | norm: {norm:.4f} | dt:{dt*1000:.2f}ms, tok/sec: {tokens_pre_sec:.2f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")