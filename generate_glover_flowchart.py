#!/usr/bin/env python3
"""
GLOVER++ 流程图生成脚本
生成包含所有模块输入输出和维度信息的详细流程图
"""

import os

def generate_mermaid_flowchart():
    """生成GLOVER++的Mermaid流程图代码"""
    
    mermaid_code = '''```mermaid
graph TD
    %% 输入数据
    A[输入数据<br/>input_dict] --> B[数据预处理]
    
    %% 数据预处理
    B --> C1[文本处理]
    B --> C2[图像处理]
    
    C1 --> D1[input_ids<br/>[batch_size, seq_len]]
    C1 --> D2[labels<br/>[batch_size, seq_len]]
    C1 --> D3[attention_masks<br/>[batch_size, seq_len]]
    C1 --> D4[offset<br/>[batch_size+1]]
    
    C2 --> E1[images<br/>[batch_size, 3, 1024, 1024]<br/>SAM输入图像]
    C2 --> E2[images_clip<br/>[batch_size, 3, 224, 224]<br/>CLIP输入图像]
    C2 --> E3[masks_list<br/>List[torch.FloatTensor]<br/>真实掩码标签]
    C2 --> E4[resize_list<br/>List[tuple]<br/>图像尺寸信息]
    
    %% 视觉特征提取
    E1 --> F[SAM Image Encoder<br/>ViT-H/14]
    F --> G[image_embeddings<br/>[batch_size, 256, 64, 64]]
    
    %% 多模态语言模型
    E2 --> H[CLIP Vision Encoder<br/>ViT-L/14]
    D1 --> I[LLaVA Model<br/>Llama-2-7B]
    D2 --> I
    D3 --> I
    H --> I
    
    I --> J[output_hidden_states<br/>[batch_size, seq_len, 4096]]
    
    %% 文本特征提取
    J --> K[text_hidden_fcs<br/>Linear Layer]
    K --> L[last_hidden_state<br/>[batch_size, seq_len, 4096]]
    
    %% [SEG]标记检测
    D1 --> M[seg_token_mask检测<br/>查找[SEG]标记位置]
    M --> N[seg_token_mask<br/>[batch_size, seq_len]<br/>布尔掩码]
    
    L --> O[提取[SEG]位置的隐藏状态]
    N --> O
    O --> P[pred_embeddings<br/>List[torch.Tensor]<br/>每个[SEG]的嵌入<br/>[num_seg_tokens, 4096]]
    
    %% 第一阶段SAM
    G --> Q1[SAM Prompt Encoder<br/>Stage 1]
    P --> Q1
    Q1 --> R1[sparse_embeddings<br/>[1, 2, 256]]
    Q1 --> S1[dense_embeddings<br/>[1, 256, 64, 64]]
    
    R1 --> T1[SAM Mask Decoder<br/>Stage 1]
    S1 --> T1
    G --> T1
    T1 --> U1[low_res_masks<br/>[1, 1, 256, 256]]
    T1 --> V1[iou_predictions<br/>[1, 1]]
    
    %% 第二阶段SAM
    U1 --> Q2[SAM Prompt Encoder<br/>Stage 2]
    P --> Q2
    Q2 --> R2[sparse_embeddings1<br/>[1, 2, 256]]
    Q2 --> S2[dense_embeddings1<br/>[1, 256, 64, 64]]
    
    R2 --> T2[SAM Mask Decoder<br/>Stage 2]
    S2 --> T2
    G --> T2
    T2 --> U2[low_res_masks1<br/>[1, 1, 256, 256]]
    T2 --> V2[iou_predictions1<br/>[1, 1]]
    
    %% 后处理
    U2 --> W[Mask Postprocessing<br/>上采样到原始尺寸]
    E4 --> W
    W --> X[pred_masks<br/>List[torch.Tensor]<br/>最终预测掩码<br/>[1, H, W]]
    
    %% 损失计算
    I --> Y1[CE Loss<br/>交叉熵损失<br/>文本生成]
    X --> Y2[Mask Loss<br/>Focal Loss + Dice Loss<br/>掩码预测]
    X --> Y3[KL Loss<br/>KL散度损失<br/>掩码质量]
    E3 --> Y2
    E3 --> Y3
    
    Y1 --> Z[Total Loss<br/>= CE + Mask + KL<br/>总损失]
    Y2 --> Z
    Y3 --> Z
    
    %% 输出
    Z --> AA[训练输出<br/>output_dict<br/>loss, ce_loss, mask_loss, kl_loss]
    X --> BB[推理输出<br/>pred_masks, gt_masks]
    
    %% 样式定义
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef lossStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef samStyle fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    %% 应用样式
    class A,B,C1,C2,D1,D2,D3,D4,E1,E2,E3,E4 inputStyle
    class F,H,I,K,M,O processStyle
    class Q1,T1,Q2,T2,W samStyle
    class AA,BB outputStyle
    class Y1,Y2,Y3,Z lossStyle
```'''
    
    return mermaid_code

def generate_html_file():
    """生成包含Mermaid流程图的HTML文件"""
    
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLOVER++ 架构流程图</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .description {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #3498db;
        }
        .description h2 {
            color: #2c3e50;
            margin-top: 0;
        }
        .description ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .description li {
            margin: 5px 0;
            line-height: 1.6;
        }
        .flowchart-container {
            text-align: center;
            margin-top: 30px;
        }
        .legend {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        .legend h3 {
            color: #2c3e50;
            margin-top: 0;
        }
        .legend-item {
            display: inline-block;
            margin: 10px 20px;
            padding: 8px 15px;
            border-radius: 5px;
            font-weight: bold;
        }
        .input-color { background-color: #e1f5fe; border: 2px solid #01579b; }
        .process-color { background-color: #f3e5f5; border: 2px solid #4a148c; }
        .sam-color { background-color: #ffebee; border: 2px solid #c62828; }
        .loss-color { background-color: #fff3e0; border: 2px solid #e65100; }
        .output-color { background-color: #e8f5e8; border: 2px solid #1b5e20; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 GLOVER++ 架构流程图</h1>
        
        <div class="description">
            <h2>📋 模型概述</h2>
            <p>GLOVER++ 是一个端到端的文本引导图像分割模型，能够根据自然语言指令在图像中定位和分割特定的交互区域（affordance）。</p>
            
            <h3>🔧 核心组件：</h3>
            <ul>
                <li><strong>多模态语言模型</strong>：基于 LLaVA (Llama-2-7B + CLIP ViT-L/14)</li>
                <li><strong>双阶段SAM</strong>：使用两个SAM模型进行串行分割</li>
                <li><strong>文本引导</strong>：通过[SEG]标记实现文本到分割的映射</li>
                <li><strong>多任务学习</strong>：同时优化文本生成和掩码预测</li>
            </ul>
            
            <h3>🎯 主要功能：</h3>
            <ul>
                <li>理解自然语言指令（如"Where should I interact with the cup?"）</li>
                <li>生成包含[SEG]标记的文本回答</li>
                <li>基于文本嵌入生成精确的分割掩码</li>
                <li>支持多种交互场景（抓取、操作、导航等）</li>
            </ul>
        </div>
        
        <div class="flowchart-container">
            <div class="mermaid">
graph TD
    %% 输入数据
    A[输入数据<br/>input_dict] --> B[数据预处理]
    
    %% 数据预处理
    B --> C1[文本处理]
    B --> C2[图像处理]
    
    C1 --> D1[input_ids<br/>[batch_size, seq_len]]
    C1 --> D2[labels<br/>[batch_size, seq_len]]
    C1 --> D3[attention_masks<br/>[batch_size, seq_len]]
    C1 --> D4[offset<br/>[batch_size+1]]
    
    C2 --> E1[images<br/>[batch_size, 3, 1024, 1024]<br/>SAM输入图像]
    C2 --> E2[images_clip<br/>[batch_size, 3, 224, 224]<br/>CLIP输入图像]
    C2 --> E3[masks_list<br/>List[torch.FloatTensor]<br/>真实掩码标签]
    C2 --> E4[resize_list<br/>List[tuple]<br/>图像尺寸信息]
    
    %% 视觉特征提取
    E1 --> F[SAM Image Encoder<br/>ViT-H/14]
    F --> G[image_embeddings<br/>[batch_size, 256, 64, 64]]
    
    %% 多模态语言模型
    E2 --> H[CLIP Vision Encoder<br/>ViT-L/14]
    D1 --> I[LLaVA Model<br/>Llama-2-7B]
    D2 --> I
    D3 --> I
    H --> I
    
    I --> J[output_hidden_states<br/>[batch_size, seq_len, 4096]]
    
    %% 文本特征提取
    J --> K[text_hidden_fcs<br/>Linear Layer]
    K --> L[last_hidden_state<br/>[batch_size, seq_len, 4096]]
    
    %% [SEG]标记检测
    D1 --> M[seg_token_mask检测<br/>查找[SEG]标记位置]
    M --> N[seg_token_mask<br/>[batch_size, seq_len]<br/>布尔掩码]
    
    L --> O[提取[SEG]位置的隐藏状态]
    N --> O
    O --> P[pred_embeddings<br/>List[torch.Tensor]<br/>每个[SEG]的嵌入<br/>[num_seg_tokens, 4096]]
    
    %% 第一阶段SAM
    G --> Q1[SAM Prompt Encoder<br/>Stage 1]
    P --> Q1
    Q1 --> R1[sparse_embeddings<br/>[1, 2, 256]]
    Q1 --> S1[dense_embeddings<br/>[1, 256, 64, 64]]
    
    R1 --> T1[SAM Mask Decoder<br/>Stage 1]
    S1 --> T1
    G --> T1
    T1 --> U1[low_res_masks<br/>[1, 1, 256, 256]]
    T1 --> V1[iou_predictions<br/>[1, 1]]
    
    %% 第二阶段SAM
    U1 --> Q2[SAM Prompt Encoder<br/>Stage 2]
    P --> Q2
    Q2 --> R2[sparse_embeddings1<br/>[1, 2, 256]]
    Q2 --> S2[dense_embeddings1<br/>[1, 256, 64, 64]]
    
    R2 --> T2[SAM Mask Decoder<br/>Stage 2]
    S2 --> T2
    G --> T2
    T2 --> U2[low_res_masks1<br/>[1, 1, 256, 256]]
    T2 --> V2[iou_predictions1<br/>[1, 1]]
    
    %% 后处理
    U2 --> W[Mask Postprocessing<br/>上采样到原始尺寸]
    E4 --> W
    W --> X[pred_masks<br/>List[torch.Tensor]<br/>最终预测掩码<br/>[1, H, W]]
    
    %% 损失计算
    I --> Y1[CE Loss<br/>交叉熵损失<br/>文本生成]
    X --> Y2[Mask Loss<br/>Focal Loss + Dice Loss<br/>掩码预测]
    X --> Y3[KL Loss<br/>KL散度损失<br/>掩码质量]
    E3 --> Y2
    E3 --> Y3
    
    Y1 --> Z[Total Loss<br/>= CE + Mask + KL<br/>总损失]
    Y2 --> Z
    Y3 --> Z
    
    %% 输出
    Z --> AA[训练输出<br/>output_dict<br/>loss, ce_loss, mask_loss, kl_loss]
    X --> BB[推理输出<br/>pred_masks, gt_masks]
    
    %% 样式定义
    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputStyle fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef lossStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef samStyle fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    %% 应用样式
    class A,B,C1,C2,D1,D2,D3,D4,E1,E2,E3,E4 inputStyle
    class F,H,I,K,M,O processStyle
    class Q1,T1,Q2,T2,W samStyle
    class AA,BB outputStyle
    class Y1,Y2,Y3,Z lossStyle
            </div>
        </div>
        
        <div class="legend">
            <h3>📊 图例说明</h3>
            <div class="legend-item input-color">输入数据</div>
            <div class="legend-item process-color">处理模块</div>
            <div class="legend-item sam-color">SAM模块</div>
            <div class="legend-item loss-color">损失计算</div>
            <div class="legend-item output-color">输出结果</div>
        </div>
        
        <div class="description">
            <h2>🔍 关键流程说明</h2>
            <ol>
                <li><strong>双路径图像处理</strong>：图像同时输入CLIP和SAM，分别用于视觉理解和分割</li>
                <li><strong>多模态融合</strong>：LLaVA模型融合文本和CLIP视觉特征</li>
                <li><strong>[SEG]标记检测</strong>：从生成的文本中检测[SEG]标记位置</li>
                <li><strong>两阶段SAM</strong>：第一阶段使用文本嵌入生成初始掩码，第二阶段使用第一阶段掩码作为提示</li>
                <li><strong>多任务损失</strong>：同时优化文本生成、掩码预测和掩码质量</li>
            </ol>
        </div>
    </div>

    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }
        });
    </script>
</body>
</html>'''
    
    return html_content

def main():
    """主函数"""
    print("🚀 正在生成GLOVER++流程图...")
    
    # 生成Mermaid代码
    mermaid_code = generate_mermaid_flowchart()
    
    # 保存Mermaid代码到文件
    with open('glover_flowchart.mmd', 'w', encoding='utf-8') as f:
        f.write(mermaid_code)
    print("✅ Mermaid代码已保存到: glover_flowchart.mmd")
    
    # 生成HTML文件
    html_content = generate_html_file()
    with open('glover_flowchart.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✅ HTML文件已保存到: glover_flowchart.html")
    
    print("\n📋 使用说明:")
    print("1. 打开 glover_flowchart.html 在浏览器中查看完整流程图")
    print("2. 复制 glover_flowchart.mmd 中的代码到 https://mermaid.live/ 进行编辑")
    print("3. 流程图包含所有模块的输入输出维度和功能说明")
    
    print("\n🎯 流程图特点:")
    print("- 完整的GLOVER++架构展示")
    print("- 详细的张量维度信息")
    print("- 清晰的模块功能说明")
    print("- 美观的样式和配色")
    print("- 支持交互式查看")

if __name__ == "__main__":
    main()




