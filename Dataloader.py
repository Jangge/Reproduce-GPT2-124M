# 将数据集处理成可以加载的程度
import tiktoken             # 导入分词工具
import torch
import os
import numpy as np

"""自定义数据集1:针对小莎士比亚文本数据集input.txt文件"""

class DataLoaderLite_input():
    
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # 从训练数据中加载数据
        with open('input.txt', 'r') as f:
            text = f.read()
        
        # 先进行分词
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)

        # 将分词后的token转为tensor格式
        self.tokens = torch.tensor(tokens)
        
        # 打印显示有多少tokens
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batchs")

        # 标记已经用过的数据
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        # 准备输入和标签
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # 下一批用新的数据
        self.current_position += B * T

        # 如果1个epoch用完了，则计数器归零
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x, y
    
"""自定义数据集2:针对fineweb-edu一共100亿tokens"""

# 加载处理好token碎片文件
def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:

    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # 获取储存tokens的碎片文件名
        data_root = "data/edu_fineweb10B"
        shards = os.listdir(data_root)              # 返回在data_root文件夹下文件名的列表
        shards = [s for s in shards if split in s]  # 根据split参数，选取是train数据集还是value数据集
        shards = sorted(shards)                     # 排序文件
        shards = [os.path.join(data_root, s) for s in shards]   # 将shards列表中的文件名加上文件夹的名字，组成路径
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        print(f"found {len(shards)} shards for split {split}")
        self.reset()                                # 为什么要调用这个函数:初始化读取文件
        
        print(f"all train_data include 100 shard, 1 shard include: {len(self.tokens)} tokens")
        print(f"1 shard = {len(self.tokens) // (self.B * self.T)} little batchs")

    # 初始化读取碎片文件
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0 

        # print(f"1 shard tokens:{len(self.tokens)}  batch:{self.B * self.T}")
        # print(f"1 shard = {len(self.tokens) // (self.B * self.T)} batchs")
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1] # 加载一个小小批量
        x = (buf[:-1]).view(B, T)                   # 训练集
        y = (buf[1:]).view(B, T)                    # 标签

        self.current_position += B * T 


        if self.current_position + (B * T  + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards) #  此处设置的特别好，是一个轮回
            self.tokens = load_tokens(self.shards[self.current_shard]) # 加载新的tokens碎片
            self.current_position = 0 
        return x, y




    


    
    
