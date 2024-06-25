"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""


import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

"""设置一些起始参数"""
local_dir = "edu_fineweb10B"         # 保存分词后文件的文件夹
remote_name = "sample-10BT"          # 要下载的数据集名称
shard_size = int(1e8)                # 每100M个token放在一个文件中，一共100个碎片文件

"""创建本地保存文件的文件夹"""
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

"""调用函数下载数据集"""
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

"""定义用于分词的超参数"""
enc = tiktoken.get_encoding("gpt2")         # 采用GPT2的格式进行分词
eot = enc._special_tokens['<|endoftext|>']  # 设定文本结束符

"""定义分词函数"""
def tokenize(doc):

    tokens = [eot]                                  # 创建一个空列表，包含一个结束符，后续分词后的token全部储存在宰割列表里 
    tokens.extend(enc.encode_ordinary(doc["text"])) # 调用函数进行分词，输入是下载的文件中字典”text"对应的值，也就是训练文本
    tokens_np = np.array(tokens)                    # 将分词后的token转换为np格式
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()  # 确保token的数量不超过2**16，这也是unit16的上限
    tokens_np_uint16 = tokens_np.astype(np.uint16)  # 将格式转换为uint16
    return tokens_np_uint16                         # 返回分词后并已经转换格式的tokens列表


"""定义将tokens写入文件的函数"""
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


if __name__ == '__main__':
    """实施分词化操作"""
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()//2)          # 选择线程，介于1和最大cpu的一般之间
    with mp.Pool(nprocs) as pool:               # 开始多线程操作
        shard_index = 0                         # 其实shard_index为0
        
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)    # 先创建一个碎片文件大小的，空的np数据
        token_count = 0                         # 用于统计写入了多少tokens的计数器
        progress_bar = None                     # 初始化进度条为0
        for tokens in pool.imap(tokenize, fw, chunksize=16):  # 这里的imap操作，是调用tokenize函数，从FW中，按照一次16个的大小抽取文件？16是指什么？

            # 判断当前的碎片文件能否继续储存读取的tokens
            if token_count + len(tokens) < shard_size:
                # 是的化，直接将tokens添加到碎片文件中
                all_tokens_np[token_count:token_count+len(tokens)] = tokens # 此处是替换空的np文件中的位置
                token_count += len(tokens)                                  # 改变计数器
                # 更新进度条
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # 第一个碎片文件为验证集，其余后面的tokens都为训练集
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                
                # 将当前的碎片空间填满，并写入文件，然后更新碎片索引并重置进度条。
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # 将当前文档中未能存入上一个碎片的词元存入新的碎片，并更新词元计数器
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # 最后，如果还有未处理完的词元（即 token_count 不为 0），将它们写入新的碎片文件
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

"""返回的碎片化数据是什么:原输入文本顺序,分词后,的np格式数据"""
