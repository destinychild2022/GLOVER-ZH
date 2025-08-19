#!/usr/bin/env python3
"""
LoRA显存占用计算器
计算不同LoRA配置下的显存占用
"""

def calculate_lora_memory(model_size_billions=7, lora_r=256, target_modules=None):
    """
    计算LoRA显存占用
    
    Args:
        model_size_billions: 模型大小（十亿参数）
        lora_r: LoRA秩
        target_modules: 目标模块列表
    """
    
    # 默认目标模块
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 估算每个模块的参数比例（基于典型Transformer架构）
    module_ratios = {
        "q_proj": 0.125,    # 12.5% of model params
        "v_proj": 0.125,    # 12.5% of model params  
        "k_proj": 0.125,    # 12.5% of model params
        "o_proj": 0.125,    # 12.5% of model params
        "gate_proj": 0.167, # 16.7% of model params
        "up_proj": 0.167,   # 16.7% of model params
        "down_proj": 0.167  # 16.7% of model params
    }
    
    # 计算总模型参数
    total_params = model_size_billions * 1e9
    
    # 计算LoRA参数
    lora_params = 0
    for module in target_modules:
        if module in module_ratios:
            module_params = total_params * module_ratios[module]
            # LoRA参数 = 2 * r * original_dim (A和B矩阵)
            lora_params += 2 * lora_r * module_params
    
    # 转换为GB (假设float16/bfloat16)
    lora_memory_gb = lora_params * 2 / (1024**3)  # 2 bytes per parameter
    
    return lora_params, lora_memory_gb

def compare_configurations():
    """比较不同LoRA配置的显存占用"""
    
    configs = [
        {"name": "保守配置", "r": 8, "modules": ["q_proj", "v_proj"]},
        {"name": "中等配置", "r": 64, "modules": ["q_proj", "v_proj", "k_proj", "o_proj"]},
        {"name": "当前配置", "r": 256, "modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        {"name": "激进配置", "r": 512, "modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    ]
    
    print("LoRA显存占用对比 (基于7B模型):")
    print("=" * 80)
    print(f"{'配置名称':<12} {'LoRA r':<8} {'目标模块数':<10} {'LoRA参数':<15} {'显存占用':<12}")
    print("-" * 80)
    
    for config in configs:
        params, memory = calculate_lora_memory(
            model_size_billions=7,
            lora_r=config["r"],
            target_modules=config["modules"]
        )
        
        print(f"{config['name']:<12} {config['r']:<8} {len(config['modules']):<10} "
              f"{params/1e6:.1f}M{'':<8} {memory:.2f}GB{'':<8}")
    
    print("\n详细分析:")
    print("1. 保守配置: 只微调attention层，显存占用最小")
    print("2. 中等配置: 微调所有attention层，平衡性能和显存")
    print("3. 当前配置: 微调所有层，显存占用较大但性能更好")
    print("4. 激进配置: 最大LoRA参数，显存占用最大")

if __name__ == "__main__":
    compare_configurations() 