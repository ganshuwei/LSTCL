'''
import pickle

# 1. 加载 .pkl 文件
with open('your_file.pkl', 'rb') as f:  # 将 'your_file.pkl' 替换为你的文件路径
    data = pickle.load(f)

# 2. 打印内容
print(data)
'''
'''
import torch
checkPoint = torch.load('/data/gsw/Code/Surgformer/results/Cholec80/surgformer_HTA_Cholec80_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4_batchSize8/checkpoint-29.pth')
print(checkPoint.keys())
'''
import numpy as np
import torch
# 加载 .npz 文件
npz_checkpoint = np.load("imagenet21k+imagenet2012_ViT-B_16-224.npz")
state_dict_npz = {key: torch.tensor(value) for key, value in npz_checkpoint.items()}
for key in list(state_dict_npz.keys()):
    if "head" in key:
        del state_dict_npz[key]
replace_rules = [
    ("Transformer/encoderblock_","blocks."),
    ("LayerNorm_0","norm1"),
    ("LayerNorm_2","norm2"),
    
]
# 查看文件中的内容
'''
with open('checkpointRecord.txt','w') as f:
    f.write("All keys:\n")
    for key in list(state_dict_npz.keys()):
        f.write(f"{key}\n")  # 列出存储在 npz 文件中的所有数组的名称
'''

