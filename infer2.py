import argparse
import os
import sys

# 在导入transformers之前禁用bitsandbytes
os.environ['BITSANDBYTES_FUNCTIONAL'] = '1'
os.environ['BITSANDBYTES_CPU_ONLY'] = '1'

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from utils.visualizer import draw_affordance, draw_affordance_center
import pdb
from torch.cuda.amp.autocast_mode import autocast


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLOVER++ chat")
    parser.add_argument("--version", default="/path/to/GLOVER++ model")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--vis_argmax_save_path", default="./vis_argmax_output", type=str
    )
    parser.add_argument("--image_path", default="", type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--objects", default="", type=str)
    parser.add_argument("--actions", default="", type=str)

    parser.add_argument(
        "--use_text_emb_in_suffix_sam", action="store_true", default=False
    )
    parser.add_argument(
        "--model_arch",
        default="glover++",
        type=str,
        choices=["glover++", "glover", "lisa"],
    )

    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=128, type=int, help="image size")  # 从256减少到128
    parser.add_argument("--model_max_length", default=64, type=int)  # 从128减少到64
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower",
        default="/path/to/clip-vit-large-patch14",
        type=str,
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def infer(args):
    print("--------------------------------------------------------")
    print("generating with model: %s..." % args.version)

    os.makedirs(args.vis_save_path, exist_ok=True)
    os.makedirs(args.vis_argmax_save_path, exist_ok=True)
    objs = args.objects.split(",")
    actions = args.actions.split(",")

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    if args.model_arch == "glover++":
        from model.GLOVER_plus import GloverForCausalLM
        model = GloverForCausalLM.from_pretrained(
            args.version,
            low_cpu_mem_usage=True,
            vision_tower=args.vision_tower,
            seg_token_idx=args.seg_token_idx,
            torch_dtype=torch_dtype
        )
    elif args.model_arch == "glover":
        from model.GLOVER import GloverForCausalLM
    model = GloverForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        torch_dtype=torch_dtype
    )
    elif args.model_arch == "lisa":
        # LISA模型使用transformers库直接加载
        from transformers import AutoModelForCausalLM
        # 设置环境变量禁用网络下载
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        model = AutoModelForCausalLM.from_pretrained(
            args.version,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            local_files_only=True  # 强制使用本地文件
        )
        # 如果模型加载返回元组，取第一个元素
        if isinstance(model, tuple):
            model = model[0]
    else:
        raise ValueError(f"Unsupported model architecture: {args.model_arch}")

    # 如果模型加载返回元组，取第一个元素
    if isinstance(model, tuple):
        model = model[0]
    
    # 启用梯度检查点来节省内存
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing for memory optimization")
    
    # 设置内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.95)  # 限制GPU内存使用为80%
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    print(f"GPU memory after model loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # 设置内存分配策略
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    # 修复设备索引问题
    device_id = 0 if args.local_rank < 0 else args.local_rank
    vision_tower.to(device=device_id)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    prompt_template = args.prompt
    image_path_template = args.image_path
    save_path_template = args.vis_save_path
    argmax_save_path_template = args.vis_argmax_save_path
    
    # 分批处理，每次只处理一个对象
    print(f"Processing {len(objs)} objects in batches...")
    
    for _, (obj, action) in enumerate(zip(objs, actions)):
        print(f"Processing {obj} - {action}")
        
        # 在处理每个对象前清理内存
        torch.cuda.empty_cache()
        print(f"GPU memory before processing {obj}: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        prompt = prompt_template % (obj, action)
        image_path = image_path_template % obj
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue
        save_path_dir = os.path.join(save_path_template, obj)
        os.makedirs(save_path_dir, exist_ok=True)
        save_path = os.path.join(save_path_dir, args.version.split("/")[-1] + ".png")

        argmax_save_path_dir = os.path.join(argmax_save_path_template, obj)
        os.makedirs(argmax_save_path_dir, exist_ok=True)
        argmax_save_path = os.path.join(
            argmax_save_path_dir, args.version.split("/")[-1] + ".png"
        )

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        if args.model_arch == "lisa":
            # LISA模型使用多模态推理，但需要特殊的tokenizer处理
            print("LISA model detected - using multimodal inference")
            
            # 处理图像
            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )

            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()

            # 对于LISA模型，使用简化的prompt以避免序列长度问题
            simple_prompt = f"Where should I interact with the {obj} to {action}?"
            print(f"Simple prompt: {simple_prompt}")
            
            # 使用tokenizer处理，并确保不超过最大长度
            input_ids = tokenizer.encode(simple_prompt, return_tensors="pt", 
                                       max_length=64, truncation=True).cuda()
            print(f"Debug: input_ids shape: {input_ids.shape}")
            print(f"Debug: input_ids max value: {input_ids.max()}")
            print(f"Debug: input_ids min value: {input_ids.min()}")
            
            # 推理
            print(f"Starting inference for {obj}...")
            print(f"Memory before inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
            with torch.no_grad():
                with autocast(enabled=True, dtype=torch.bfloat16):
                    # 使用模型的forward方法而不是generate
                    model_inputs = {
                        'input_ids': input_ids,
                        'images': image_clip,
                        'return_dict': True
                    }
                    
                    # 先进行一次forward pass
                    outputs = model(**model_inputs)
                    
                    # 然后手动生成文本
                    logits = outputs.logits
                    next_token_logits = logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                    
                    # 生成完整的输出序列
                    generated_ids = input_ids.clone()
                    for _ in range(32):  # 最多生成32个token
                        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                        
                        # 检查是否生成了结束token
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                            
                        # 继续生成下一个token
                        model_inputs['input_ids'] = generated_ids
                        outputs = model(**model_inputs)
                        logits = outputs.logits
                        next_token_logits = logits[0, -1, :]
                        next_token = torch.argmax(next_token_logits).unsqueeze(0)
            
            # LISA模型不返回分割掩码，只返回文本
            output_ids = generated_ids
            pred_masks = []  # LISA模型不支持分割
        else:
            # GLOVER模型使用多模态推理
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )

        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # 推理
        print(f"Starting inference for {obj}...")
        print(f"Memory before inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
            # GLOVER模型使用自定义evaluate方法
        with torch.no_grad():
            with autocast(enabled=True, dtype=torch.bfloat16):
                    outputs = model.evaluate(
                    image_clip,
                    image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=32,  # 减少到32
                    tokenizer=tokenizer,
                    **(
                        {"use_text_emb_in_suffix_sam": args.use_text_emb_in_suffix_sam}
                        if args.model_arch == "glover++"
                        else {}
                    )
                )
            
            # 正确处理输出
            if hasattr(outputs, 'sequences'):
                output_ids = outputs.sequences
            else:
                output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
                
            if hasattr(outputs, 'pred_masks'):
                pred_masks = outputs.pred_masks
            else:
                pred_masks = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else []
        
        print(f"Memory after inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output for %s: %s" % (obj, text_output))

        # 处理分割掩码和保存图片
        if args.model_arch == "lisa":
            # LISA模型不支持分割，但我们可以保存原始图像和文本输出
            print(f"LISA model: Saving original image and text output for {obj}")
            
            # 保存原始图像
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_np)
            
            # 保存文本输出到文件
            text_save_path = save_path.replace('.png', '_text.txt')
            with open(text_save_path, 'w') as f:
                f.write(f"Object: {obj}\n")
                f.write(f"Action: {action}\n")
                f.write(f"Model: LISA_Plus_7b\n")
                f.write(f"Text Output: {text_output}\n")
            
            print(f"Saved original image to: {save_path}")
            print(f"Saved text output to: {text_save_path}")
        else:
            # GLOVER模型处理分割掩码
        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask[0].sigmoid()
            pred_mask = pred_mask.detach().cpu().numpy()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            aff_vis, _ = draw_affordance(image_np, pred_mask)
            cv2.imwrite(save_path, aff_vis)
            aff_vis_center = draw_affordance_center(image_np, pred_mask)
            cv2.imwrite(argmax_save_path, aff_vis_center)

        # 彻底清理内存
        if args.model_arch == "lisa":
            del image_clip, input_ids, image_np
        else:
        del image_clip, image, input_ids, image_np
        torch.cuda.empty_cache()
        print(f"GPU memory after processing {obj}: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    infer(args)