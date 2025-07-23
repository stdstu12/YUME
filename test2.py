import os
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def split_safetensors(input_path, output_dir, num_parts=7):
    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 获取基础文件名
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 3. 加载原始模型的张量和元数据
    with safe_open(input_path, framework="pt") as f:
        all_tensors = {}
        all_keys = []
        for key in f.keys():
            all_tensors[key] = f.get_tensor(key)
            all_keys.append(key)
    
    # 4. 将键分成num_parts组
    key_groups = [all_keys[i::num_parts] for i in range(num_parts)]
    
    # 5. 创建并保存元数据文件
    index_data = {
        "metadata": {"total_parts": num_parts},
        "weight_map": {}
    }
    
    # 6. 为每个部分创建safetensors文件
    for i, keys in enumerate(key_groups):
        # 保存部分文件
        part_filename = f"{base_name}-{i+1:05d}-of-{num_parts:05d}.safetensors"
        part_path = os.path.join(output_dir, part_filename)
        save_file({k: all_tensors[k] for k in keys}, part_path)
        
        # 更新索引映射
        for key in keys:
            index_data["weight_map"][key] = part_filename
    
    # 7. 保存索引文件
    index_path = os.path.join(output_dir, f"{base_name}.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    
    print(f"成功将 {input_path} 拆分成 {num_parts} 个部分")

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际路径
    input_model = "/mnt/petrelfs/maoxiaofeng/Yume_v1_release/Yume-I2V-540P/Yume-Dit/diffusion_pytorch_model.safetensors"
    output_directory = "/mnt/petrelfs/maoxiaofeng/Yume_v1_release/Yume-I2V-540P/Yume-Dit"
    
    split_safetensors(input_model, output_directory, num_parts=7)