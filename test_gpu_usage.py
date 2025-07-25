import torch
import os

def test_gpu_usage():
    print("=== GPU Usage Test ===")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device before setting: {torch.cuda.current_device()}")
    
    # 设置使用GPU 2和3
    if torch.cuda.device_count() > 3:
        device_ids = [2, 3]
        device = torch.device("cuda:2")
        torch.cuda.set_device(2)
        print(f"Using multiple GPUs: {device_ids}")
        print(f"Current device after setting: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # 测试DataParallel
        from torch.nn import DataParallel
        print("\nTesting DataParallel...")
        
        # 创建一个简单的测试模型
        test_model = torch.nn.Linear(10, 10)
        test_model = test_model.to(device)
        
        # 包装为DataParallel
        dp_model = DataParallel(test_model, device_ids=[2, 3])
        print(f"DataParallel model device_ids: {dp_model.device_ids}")
        print(f"DataParallel model module device: {next(dp_model.parameters()).device}")
        
        # 测试输入
        test_input = torch.randn(4, 10).to(device)
        output = dp_model(test_input)
        print(f"Output device: {output.device}")
        print(f"Output shape: {output.shape}")
        
    else:
        print("Not enough GPUs for multi-GPU testing")

if __name__ == "__main__":
    test_gpu_usage() 