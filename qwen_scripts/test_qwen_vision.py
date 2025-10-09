#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import argparse

def test_qwen_vision():
    """测试Qwen模型的基础视觉识别能力"""
    
    print("=" * 60)
    print("测试Qwen模型的基础视觉识别能力")
    print("=" * 60)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型和tokenizer
    model_path = "/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL"
    
    print(f"加载模型: {model_path}")
    
    # 使用量化配置以节省内存
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            local_files_only=True
        )
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试图片路径
    test_image_path = "/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images/book.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图片不存在: {test_image_path}")
        return
    
    print(f"测试图片: {test_image_path}")
    
    # 加载图片
    try:
        image = Image.open(test_image_path).convert("RGB")
        print(f"✅ 图片加载成功，尺寸: {image.size}")
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return
    
    # 测试问题列表
    test_questions = [
        "图片中有几本书？",
        "图片中有什么物体？",
        "请描述这张图片的内容。",
        "图片中的书本是什么颜色的？",
        "图片中有多少个物体？"
    ]
    
    print("\n" + "=" * 60)
    print("开始测试视觉识别能力")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 40)
        
        try:
            # 构建对话
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # 应用对话模板
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 编码输入
            image_inputs, video_inputs = model.process_vision_info(messages)
            inputs = tokenizer(
                text,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            )
            
            # 生成回答
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=50,
                    repetition_penalty=1.1
                )
            
            # 解码回答
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"回答: {response}")
            
        except Exception as e:
            print(f"❌ 生成回答失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_qwen_vision()
