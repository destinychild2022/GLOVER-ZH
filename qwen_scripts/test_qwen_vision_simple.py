#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def test_qwen_vision():
    """测试Qwen2.5-VL模型的基础视觉识别能力"""
    
    print("=" * 60)
    print("测试Qwen2.5-VL模型的基础视觉识别能力")
    print("=" * 60)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 模型路径
    model_path = "/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL"
    
    print(f"加载模型: {model_path}")
    
    try:
        # 加载模型和处理器
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto",
            local_files_only=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试图片路径
    test_image_path = "/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images/book.png"
    
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图片不存在: {test_image_path}")
        return
    
    print(f"测试图片: {test_image_path}")
    
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
            # 构建对话消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{test_image_path}"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # 准备推理
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # 生成回答
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            print(f"回答: {output_text[0]}")
            
        except Exception as e:
            print(f"❌ 生成回答失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_qwen_vision()

