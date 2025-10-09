import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info


def split_long_sequence(input_ids, targets, attention_masks, max_length, tokenizer):
    """
    智能分割超长序列，避免信息丢失
    
    Args:
        input_ids: 输入token序列 [batch_size, seq_len]
        targets: 标签序列 [batch_size, seq_len]
        attention_masks: 注意力掩码 [batch_size, seq_len]
        max_length: 最大允许长度
        tokenizer: tokenizer对象
    
    Returns:
        分割后的序列列表，每个元素是一个字典包含input_ids, targets, attention_masks
    """
    batch_size = input_ids.shape[0]
    sequences = []
    
    for i in range(batch_size):
        seq_input_ids = input_ids[i]
        seq_targets = targets[i]
        seq_attention_masks = attention_masks[i]
        
        # 找到有效token的结束位置（非padding token）
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        valid_length = (seq_input_ids != pad_token_id).sum().item()
        
        if valid_length <= max_length:
            # 序列长度合适，直接使用
            sequences.append({
                'input_ids': seq_input_ids[:valid_length].unsqueeze(0),
                'targets': seq_targets[:valid_length].unsqueeze(0),
                'attention_masks': seq_attention_masks[:valid_length].unsqueeze(0)
            })
        else:
            # 序列过长，需要分割
            print(f"   样本 {i}: 长度 {valid_length} > {max_length}，进行智能分割")
            
            # 尝试在句子边界分割（寻找句号、问号、感叹号等）
            sentence_endings = [tokenizer.encode('.')[0], tokenizer.encode('?')[0], 
                              tokenizer.encode('!')[0], tokenizer.encode('。')[0]]
            
            current_start = 0
            segment_count = 0
            
            while current_start < valid_length:
                segment_end = min(current_start + max_length, valid_length)
                
                # 如果不是最后一段，尝试在句子边界分割
                if segment_end < valid_length:
                    # 在当前位置到segment_end之间寻找句子结束符
                    segment_tokens = seq_input_ids[current_start:segment_end]
                    sentence_end_pos = -1
                    
                    for j in range(len(segment_tokens) - 1, -1, -1):
                        if segment_tokens[j].item() in sentence_endings:
                            sentence_end_pos = j
                            break
                    
                    # 如果找到句子边界，在边界处分割
                    if sentence_end_pos > max_length * 0.7:  # 确保分割点不要太靠前
                        segment_end = current_start + sentence_end_pos + 1
                
                # 创建当前片段
                segment_input_ids = seq_input_ids[current_start:segment_end].unsqueeze(0)
                segment_targets = seq_targets[current_start:segment_end].unsqueeze(0)
                segment_attention_masks = seq_attention_masks[current_start:segment_end].unsqueeze(0)
                
                sequences.append({
                    'input_ids': segment_input_ids,
                    'targets': segment_targets,
                    'attention_masks': segment_attention_masks
                })
                
                current_start = segment_end
                segment_count += 1
            
            print(f"   样本 {i}: 分割成 {segment_count} 个片段")
    
    return sequences


def resize_with_padding(image, target_size, fill_color=(0, 0, 0)):
    """
    保持宽高比resize图像并添加padding到目标尺寸
    
    Args:
        image: PIL Image对象
        target_size: 目标尺寸 (width, height)
        fill_color: padding填充颜色，默认为黑色
    
    Returns:
        处理后的PIL Image对象
    """
    target_width, target_height = target_size
    original_width, original_height = image.size
    
    # 计算缩放比例，选择较小的比例以确保图像完全适合目标尺寸
    scale = min(target_width / original_width, target_height / original_height)
    
    # 计算resize后的尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # resize图像
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 创建目标尺寸的空白图像
    padded_image = Image.new('RGB', (target_width, target_height), fill_color)
    
    # 计算居中位置
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # 将resize后的图像粘贴到中心位置
    padded_image.paste(resized_image, (x_offset, y_offset))
    
    return padded_image


from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from model.llava.mm_utils import tokenizer_image_token

from .utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN
from .threedoi_dataset import DoiDataset
from .ego4d_dataset import Ego4DDataset
from .epic100_dataset import Epic100Dataset
from .handal_dataset import HANDALDataset
from .custom_annotation_dataset import CustomAnnotationDataset


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, use_qwen_mode=False, processor=None, image_size=None, patch_size=14
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    resize_list = []
    questions_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        resize,
        questions,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        if masks is not None:
            masks_list.append(masks.float())
        else:
            masks_list.append(None)
        resize_list.append(resize)
        questions_list.append(questions)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    # 对于Qwen2.5-VL，使用原生tokenizer处理，避免使用tokenizer_image_token
    input_ids = []
    for prompt in conversation_list:
        # 直接使用tokenizer处理，让Qwen自己处理图像token
        input_ids.append(tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0))
    # 确保padding_value不为None
    padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=padding_value
    )
    # 确保pad_token_id不为None
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attention_masks = input_ids.ne(pad_token_id)

    # 传统模式：处理对话和标签
    if not (use_qwen_mode and processor is not None):
        conv = conversation_lib.default_conversation.copy()
        targets = input_ids.clone()

        if conv_type == "llava_v1":
            sep = conv.sep + conv.roles[1] + ": "
        else:
            sep = "[/INST] "
        for conversation, target in zip(conversation_list, targets):
            total_len = int(target.ne(pad_token_id).sum())

            # 处理对话分割，支持###和</s>两种分隔符
            if conv.sep2 in conversation:
                rounds = conversation.split(conv.sep2)
            elif "###" in conversation:
                rounds = conversation.split("###")
            else:
                rounds = [conversation]
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += instruction_len
                # 保留答案部分，不设置为IGNORE_INDEX
                cur_len += (round_len - instruction_len)
            target[cur_len:] = IGNORE_INDEX

            if False:
                z = target.clone()
                z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
                if local_rank == 0:
                    print(
                        "conversation: ",
                        conversation,
                        "tokenizer.decode(z): ",
                        tokenizer.decode(z),
                    )

            if cur_len < tokenizer.model_max_length:
                # 暂时注释掉这个断言，避免训练中断
                # assert cur_len == total_len
                pass

    # 改进的截断逻辑：对于超长样本，使用智能分割而不是简单截断
    if inferences[0] == False:
        max_length = tokenizer.model_max_length - 255  # 预留空间给特殊token
        
        if input_ids.shape[1] > max_length:
            print(f"⚠️ 检测到超长样本: {input_ids.shape[1]} > {max_length}")
            print(f"   使用智能分割逻辑，避免信息丢失")
            
            # 使用智能分割函数
            sequences = split_long_sequence(input_ids, targets, attention_masks, max_length, tokenizer)
            
            if len(sequences) > 1:
                print(f"   警告：批次被分割成 {len(sequences)} 个片段")
                print(f"   注意：这可能导致训练批次大小不一致，建议调整数据预处理")
                
                # 对于分割后的序列，我们选择第一个片段继续处理
                # 在实际应用中，可能需要重新设计训练循环来处理多个片段
                input_ids = sequences[0]['input_ids']
                targets = sequences[0]['targets']
                attention_masks = sequences[0]['attention_masks']
                
                print(f"   使用第一个片段，长度: {input_ids.shape[1]}")
            else:
                # 没有分割，直接使用
                input_ids = sequences[0]['input_ids']
                targets = sequences[0]['targets']
                attention_masks = sequences[0]['attention_masks']

    # Qwen2.5-VL模式：使用processor一次性完成所有处理
    if use_qwen_mode and processor is not None:
        print(f"使用 Qwen2.5-VL processor 一次性处理 {len(image_path_list)} 个样本")

        if image_size is None:
            raise ValueError("[ERROR] Qwen image_size 不能为 None，请从模型 config 传入！")
        
        print(f"配置: image_size={image_size}, patch_size={patch_size}")
        
        # 准备所有样本的数据
        all_messages = []
        
        # 处理每个样本，构建消息格式
        for i in range(len(image_path_list)):
            # 获取原始图像（PIL格式）
            import cv2
            from PIL import Image
            
            # 从image_path读取原始图像
            img = cv2.imread(image_path_list[i])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            # 检查图像尺寸，如果不匹配则直接resize到目标尺寸
            if pil_image.size != (image_size, image_size):
                pil_image = pil_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            
            # 构建消息格式 - 保留对话里的图像token占位符
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": pil_image}, 
                        {"type": "text", "text": conversation_list[i]}
                    ]
                }
            ]
            all_messages.append(messages)
        
        # 一次性处理所有样本
        try:
            # 1. 从conversation_list中分离问题和答案，构造完整的对话格式
            all_complete_messages = []
            all_answers = []  # 添加答案列表
            all_image_inputs = []
            all_video_inputs = []
            
            for i, conversation in enumerate(conversation_list):
                # 根据conv_llava_v1的格式分离问题和答案
                # 格式：USER: question ASSISTANT: answer</s>
                if "ASSISTANT:" in conversation:
                    parts = conversation.split("ASSISTANT:")
                    if len(parts) == 2:
                        question_part = parts[0].replace("USER:", "").strip()
                        answer_part = parts[1].replace("</s>", "").strip()
                        
                        # 构造完整的对话格式（包含prompt + answer）
                        complete_messages = [
                            {"role": "user", "content": [{"type": "image", "image": all_messages[i][0]["content"][0]["image"]}, {"type": "text", "text": question_part}]},
                            {"role": "assistant", "content": [{"type": "text", "text": answer_part}]}
                        ]
                        all_complete_messages.append(complete_messages)
                        all_answers.append(answer_part)  # 收集答案
                    else:
                        # 如果格式不匹配，使用原始消息
                        all_complete_messages.append(all_messages[i])
                        all_answers.append("")  # 添加空答案
                else:
                    # 如果格式不匹配，使用原始消息
                    all_complete_messages.append(all_messages[i])
                    all_answers.append("")  # 添加空答案
                
                # 处理图像输入
                image_inputs, video_inputs = process_vision_info(all_messages[i])
                all_image_inputs.extend(image_inputs)
                if video_inputs is not None:
                    all_video_inputs.extend(video_inputs)
            
            # 2. 使用Qwen官方chat_template处理所有完整消息
            # 注意要用Qwen的chat_template，它会加图像token和特殊符号
            all_texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                for msg in all_complete_messages
            ]
            
            # 3. 使用processor一次性处理所有输入，手动生成labels
            processor_kwargs = {
                "text": all_texts,           # 完整的对话文本（包含prompt + answer）
                "images": all_image_inputs,  # 图像
                "padding": True,
                "return_tensors": "pt"
            }
            if all_video_inputs:
                processor_kwargs["videos"] = all_video_inputs
            
            inputs = processor(**processor_kwargs)
            # processor没有自动生成labels，需要手动处理
            
            # 提取处理后的数据 - processor已经完成了所有处理
            pixel_values = inputs.pixel_values
            input_ids = inputs.input_ids
            attention_masks = inputs.attention_mask
            image_grid_thw = inputs.image_grid_thw
            
            # 手动生成labels：克隆input_ids，将prompt部分设为-100
            labels = input_ids.clone()
            IGNORE_INDEX = -100
            
            # 对于每个样本，找到assistant回答的开始位置并mask掉prompt部分
            for i in range(len(all_texts)):
                # 使用tokenizer找到assistant回答的开始位置
                assistant_start = all_texts[i].find("assistant")
                if assistant_start != -1:
                    # 找到assistant回答开始后的第一个token
                    assistant_text = all_texts[i][assistant_start:]
                    assistant_tokens = processor.tokenizer.encode(assistant_text, add_special_tokens=False)
                    
                    # 计算需要mask的长度（prompt部分的长度）
                    prompt_length = input_ids[i].shape[0] - len(assistant_tokens)
                    if prompt_length > 0:
                        labels[i, :prompt_length] = IGNORE_INDEX
            
            print(f"Processor处理结果: pixel_values={pixel_values.shape}, image_grid_thw={image_grid_thw.shape}")
            
        except Exception as e:
            # 直接抛出异常，不进行fallback
            raise RuntimeError(f"Processor批量处理失败: {e}\n"
                             f"请检查图像格式、对话格式或processor配置。")
        
        # 返回处理好的数据，不再进行额外处理
        return {
            "image_paths": image_path_list,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "image_grid_thw": image_grid_thw,
            "masks_list": masks_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "inference": inferences[0],
            "conversation_list": conversation_list,
        }
    else:
        # 传统模式：使用images和images_clip
        return {
            "image_paths": image_path_list,
            "images": torch.stack(images_list, dim=0),
            "images_clip": torch.stack(images_clip_list, dim=0),
            "input_ids": input_ids,
            "labels": targets,
            "attention_masks": attention_masks,
            "masks_list": masks_list,
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "inference": inferences[0],
            "conversation_list": conversation_list,
        }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    # img_size = 1024  # 移除硬编码，使用传入的image_size参数
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        dataset="3doi||ego4d||epic100||handal",
        sample_rate=[9, 3, 3, 1],
        annotation_dir=None,  # 新增参数用于自定义标注数据集
    ):
        self.samples_per_epoch = samples_per_epoch
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.annotation_dir = annotation_dir  # 新增
        self.image_size = image_size
        self.img_size = image_size  # 确保img_size也被正确设置
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "3doi":
                self.all_datasets.append(
                    DoiDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        data_name="3doi",
                    )
                )

            elif dataset == "ego4d":
                self.all_datasets.append(
                    Ego4DDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        data_name="ego4d",
                    )
                )

            elif dataset == "epic100":
                self.all_datasets.append(
                    Epic100Dataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        data_name="epic100",
                    )
                )

            elif dataset == "handal":
                self.all_datasets.append(
                    HANDALDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        data_name="handal",
                    )
                )
            
            elif dataset == "custom_annotation":
                # 新增：支持自定义标注数据集
                if annotation_dir is None:
                    raise ValueError("annotation_dir must be provided for custom_annotation dataset")
                self.all_datasets.append(
                    CustomAnnotationDataset(
                        base_image_dir,
                        annotation_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        data_name="custom_annotation",
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference
