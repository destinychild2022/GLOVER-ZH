from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BitsAndBytesConfig, CLIPVisionModel, PreTrainedModel, GenerationMixin, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava.model.language_model.llava_qwen import (
    LlavaQwenForCausalLM,
    LlavaQwenModel,
)
from .segment_anything import build_sam_vit_h
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
import numpy as np
import pdb


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def cal_kl(inputs, targets, eps=1e-12):
    """
    计算KL散度，与GLOVER_plus.py完全一致
    """
    inputs = inputs.sigmoid()
    inputs = inputs / (inputs.sum(dim=(1, 2), keepdim=True) + eps)
    targets = targets / (targets.sum(dim=(1, 2), keepdim=True) + eps)

    kld = targets * (torch.log(targets + eps) - torch.log(inputs + eps))
    kld = kld.sum()

    return kld


class GloverMetaModel(nn.Module):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # 作为nn.Module注册所有子模块
        nn.Module.__init__(self)
        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs.get("train_mask_decoder", True)
            self.config.out_dim = kwargs.get("out_dim", 256)
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            # 延迟初始化，等模型加载完成后再调用
            self._glover_initialized = False

    def _ensure_glover_initialized(self):
        """确保GLOVER模块已初始化"""
        if not hasattr(self, '_glover_initialized') or not self._glover_initialized:
            self.initialize_glover_modules(self.config)
            self._glover_initialized = True

    def initialize_glover_modules(self, config):
        # 检查是否是Qwen2.5-VL模型
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'visual'):
            print("检测到Qwen2.5-VL模型，使用内置视觉编码器")
            # 对于Qwen2.5-VL，使用其内置的视觉编码器
            self.visual_model = self.model.model.visual
            # 创建SAM模型用于分割任务，按照原始GLOVER的方式
            self.visual_model1 = build_sam_vit_h(self.vision_pretrained)
            # 冻结Qwen2.5-VL的视觉编码器
            for param in self.visual_model.parameters():
                param.requires_grad = False
                
            # 训练SAM的mask decoder
            if config.train_mask_decoder:
                self.visual_model1.mask_decoder.train()
                for param in self.visual_model1.mask_decoder.parameters():
                    param.requires_grad = True
        else:
            print("使用传统SAM模型")
            # 传统SAM模型
            self.visual_model = build_sam_vit_h(self.vision_pretrained)
            self.visual_model1 = build_sam_vit_h(self.vision_pretrained)
            for param in self.visual_model.parameters():
                param.requires_grad = False
            if config.train_mask_decoder:
                self.visual_model.mask_decoder.train()
                for param in self.visual_model.mask_decoder.parameters():
                    param.requires_grad = True
        
        # 全局设置SAM模型的精度和设备管理
        self._setup_sam_precision_and_device(config)

    def _setup_sam_precision_and_device(self, config):
        """全局设置SAM模型的精度和设备管理，避免forward中频繁.to()调用"""
        print(f"[DEBUG] 设置SAM模型精度和设备管理")
        
        # 获取目标精度
        target_dtype = getattr(config, 'torch_dtype', torch.float32)
        if hasattr(self.model, 'dtype'):
            target_dtype = self.model.dtype
        
        # 如果模型有precision属性，使用它来确定精度
        if hasattr(self, 'precision'):
            if self.precision == 'bf16':
                target_dtype = torch.bfloat16
            elif self.precision == 'fp16':
                target_dtype = torch.half
            else:
                target_dtype = torch.float32
        
        print(f"[DEBUG] 目标精度: {target_dtype}")
        
        # 设置SAM模型的精度和设备（只对visual_model1，即SAM模型）
        if hasattr(self, 'visual_model1'):
            print(f"[DEBUG] 设置visual_model1（SAM模型）精度和设备")
            
            # 获取当前设备（通常是GPU）
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cuda:0')
            print(f"[DEBUG] 将SAM模型移动到设备: {device}")
            
            # 对于image_encoder和prompt_encoder，移动到GPU并使用目标精度
            self.visual_model1.image_encoder = self.visual_model1.image_encoder.to(device=device, dtype=target_dtype)
            self.visual_model1.prompt_encoder = self.visual_model1.prompt_encoder.to(device=device, dtype=target_dtype)
            
            # 对于mask_decoder，移动到GPU并使用与训练一致的精度
            print(f"[DEBUG] mask_decoder使用精度: {target_dtype}, 设备: {device}")
            self.visual_model1.mask_decoder = self.visual_model1.mask_decoder.to(device=device, dtype=target_dtype)
            
            # 标记已设置精度，避免forward中重复设置
            self._sam_precision_set = True
            self._sam_target_dtype = target_dtype
            print(f"[DEBUG] SAM精度设置完成")
        
        # 注意：visual_model 是 Qwen2.5-VL 的视觉编码器，不是SAM结构
        # 不需要设置 image_encoder, prompt_encoder, mask_decoder 等SAM属性
        if hasattr(self, 'visual_model') and self.visual_model != self.visual_model1:
            print(f"[DEBUG] visual_model 是 Qwen2.5-VL 视觉编码器，无需设置SAM相关精度")

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]

        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class GloverQwenModel(GloverMetaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GloverQwenModel, self).__init__(config, **kwargs)

        # 根据配置类型选择合适的模型类
        if hasattr(config, 'model_type') and config.model_type in ['qwen2_vl', 'qwen2_5_vl']:
            # 使用Qwen2-VL模型，直接从预训练权重加载

            # 直接从预训练路径加载，使用原始配置
            model_path = config._name_or_path if hasattr(config, '_name_or_path') else None
            if model_path:
                print(f"正在从预训练路径加载模型: {model_path}")
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map=None  # 让DeepSpeed处理设备映射
                )
                print("✅ 成功加载预训练权重")
            else:
                raise ValueError("没有找到预训练模型路径")
            # except Exception as e:
            #     print(f"加载Qwen2-VL预训练模型失败: {e}")
            #     # 回退到从配置创建，使用AutoModel
            #     print("使用随机初始化的模型")
            #     from transformers import AutoModel
            #     self.model = AutoModel.from_config(config, trust_remote_code=True)
        # else:
        #     # 使用普通的Qwen2模型
        #     from transformers import AutoModelForCausalLM
        #     self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        # # 为了PEFT兼容性，添加prepare_inputs_for_generation方法
        # if not hasattr(self.model, 'prepare_inputs_for_generation'):
        #     self.model.prepare_inputs_for_generation = self._prepare_inputs_for_generation
        
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False
        
        # 确保vision_pretrained被设置
        if not hasattr(self, 'vision_pretrained'):
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        
        # 初始化GLOVER的视觉组件
        self.initialize_glover_modules(config)



    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        return self.model.gradient_checkpointing_disable()
        
    def get_vision_tower(self):
        return getattr(self, "vision_tower", None)
        
    def embed_tokens(self, input_ids):
        # Qwen模型使用get_input_embeddings方法
        embeddings = self.model.get_input_embeddings()
        
        # 确保输入在正确的设备上
        device = next(embeddings.parameters()).device
        input_ids = input_ids.to(device)
        
        # 获取嵌入
        result = embeddings(input_ids)
        
        # 确保返回的张量与lm_head所在设备一致（ZeRO下参数可能在CPU，不能用任意参数判断）
        target_device = (
            self.model.lm_head.weight.device
            if hasattr(self.model, "lm_head") and hasattr(self.model.lm_head, "weight")
            else self.lm_head.weight.device
        )
        return result.to(target_device)
    
    def _prepare_inputs_for_generation(self, *args, **kwargs):
        """为了PEFT兼容性添加的方法"""
        return self.model._prepare_inputs_for_generation(*args, **kwargs) if hasattr(self.model, '_prepare_inputs_for_generation') else None
        
    def initialize_vision_modules(self, model_args, fsdp=None):
        """初始化视觉模块，从LlavaMetaModel复制"""
        # 检查是否是Qwen2.5-VL模型
        if hasattr(self, 'model') and hasattr(self.model, 'visual'):
            print("检测到Qwen2.5-VL模型，跳过外部视觉编码器初始化")
            print("使用Qwen2.5-VL内置的视觉编码器")
            
            # 设置必要的配置
            self.config.use_mm_proj = True
            self.config.mm_hidden_size = self.model.visual.config.hidden_size
            self.config.mm_vision_select_layer = -2  # 使用倒数第二层
            self.config.mm_vision_select_feature = "patch"
            
            # 对于Qwen2.5-VL，不需要外部的mm_projector，使用内置的
            if getattr(self, "mm_projector", None) is None:
                print("使用Qwen2.5-VL内置的多模态投影器")
                # Qwen2.5-VL的投影器已经内置在模型中
                self.mm_projector = None
            else:
                # In case it is frozen by LoRA
                for p in self.mm_projector.parameters():
                    p.requires_grad = True
            
            return
        
        # 对于非Qwen2.5-VL模型，使用原来的逻辑
        print("使用外部CLIP视觉编码器")
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        from .llava.model.multimodal_encoder.builder import build_vision_tower
        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        from .llava.model.multimodal_projector.builder import build_vision_projector
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class GloverQwenForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = None  # 我们将使用AutoConfig
    supports_gradient_checkpointing = True
    
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def _ensure_dtype_consistency(self, tensor, target_module):
        """
        确保张量与目标模块权重dtype一致，解决混合精度训练中的dtype不匹配问题
        添加数值稳定性检查，防止NaN传播
        特殊处理SAM模型的fp32精度
        """
        if target_module is None or tensor is None:
            return tensor
        
        # 检查输入张量是否包含NaN或Inf
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"[WARNING] 输入张量包含NaN或Inf，形状: {tensor.shape}")
            # 用零替换NaN和Inf
            tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                               torch.zeros_like(tensor), tensor)
        
        try:
            target_dtype = next(target_module.parameters()).dtype
            
            # 特殊处理：如果目标模块是SAM模型，使用与模型相同的精度
            if 'sam' in str(type(target_module)).lower() or 'mask_decoder' in str(type(target_module)).lower():
                # 在推理时，SAM模型使用与主模型相同的精度
                target_dtype = next(target_module.parameters()).dtype
                print(f"[INFO] SAM模型使用{target_dtype}精度")
            
            converted_tensor = tensor.to(target_dtype)
            
            # 检查转换后是否产生NaN
            if torch.isnan(converted_tensor).any() or torch.isinf(converted_tensor).any():
                print(f"[WARNING] 类型转换后产生NaN或Inf，目标dtype: {target_dtype}")
                # 用零替换NaN和Inf
                converted_tensor = torch.where(torch.isnan(converted_tensor) | torch.isinf(converted_tensor), 
                                             torch.zeros_like(converted_tensor), converted_tensor)
            
            return converted_tensor
        except (StopIteration, AttributeError):
            # 如果模块没有参数或无法获取dtype，返回原张量
            return tensor

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """获取视觉嵌入，支持Qwen2.5-VL和传统SAM模型"""
        with torch.no_grad():
            # 直接使用SAM模型进行视觉编码，避免复杂的Qwen2.5-VL视觉编码器调用
            if hasattr(self.model, 'visual_model1') and hasattr(self.model.visual_model1, 'image_encoder'):
                # 确保SAM模型在正确的设备上
                target_device = pixel_values.device
                target_dtype = pixel_values.dtype
                
                # 移动SAM模型到目标设备
                self.model.visual_model1.image_encoder = self.model.visual_model1.image_encoder.to(device=target_device, dtype=target_dtype)
                
                # 确保pixel_values在正确的设备和数据类型上
                if pixel_values.device != target_device:
                    pixel_values = pixel_values.to(target_device)
                if pixel_values.dtype != target_dtype:
                    pixel_values = pixel_values.to(target_dtype)
                
                image_embeddings = self.model.visual_model1.image_encoder(pixel_values)
                
                # 确保返回的image_embeddings在正确的设备上
                if image_embeddings.device != target_device:
                    image_embeddings = image_embeddings.to(device=target_device, dtype=target_dtype)
            else:
                raise RuntimeError("无法获取视觉嵌入")
        return image_embeddings

    def encode_images(self, images):
        # 对于Qwen2.5-VL，使用内置的视觉编码器
        if hasattr(self.get_model(), 'model') and hasattr(self.get_model().model, 'visual'):
            # 使用Qwen2.5-VL内置的视觉编码器
            vision_model = self.get_model().model.visual
            image_features = vision_model(images)
            return image_features
        else:
            # 使用传统的CLIP视觉编码器
            vision_tower = self.get_model().get_vision_tower()
            if vision_tower is None:
                raise RuntimeError("没有可用的视觉编码器")
            
            image_features = vision_tower(images)
            # 与mm_projector权重设备对齐特征，避免在前向中移动模块
            mm_projector = self.get_model().mm_projector
            if mm_projector is not None:
                target_device = mm_projector.weight.device
                if image_features.device != target_device:
                    image_features = image_features.to(target_device)
                image_features = mm_projector(image_features)
            return image_features

    def generate(self, input_ids, images=None, **kwargs):
        """生成文本的方法，支持多模态输入，使用LISA的embedding-as-mask范式"""
        # 如果有images参数，先处理多模态输入
        if images is not None:
            # 处理多模态输入，将图像特征融入到输入中
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=kwargs.get('attention_mask'),
                past_key_values=kwargs.get('past_key_values'),
                labels=kwargs.get('labels'),
                images=images
            )
            kwargs['inputs_embeds'] = inputs_embeds
            if attention_mask is not None:
                kwargs['attention_mask'] = attention_mask
            # 移除images参数，因为父类不支持
            kwargs.pop('images', None)
        
        # 移除强制偏置，让模型自然学习生成[SEG] token
        
        # 使用 transformers 的生成方法
        from transformers import GenerationMixin
        return GenerationMixin.generate(self, input_ids, **kwargs)
        
    def evaluate(self, image_clip, image, input_ids, resize_list, original_size_list, 
                max_new_tokens=512, tokenizer=None, use_text_emb_in_suffix_sam=False, image_grid_thw=None, **kwargs):
        """GLOVER模型的评估方法，生成分割掩码和文本输出"""
        with torch.no_grad():
            # 手动检查input_ids中是否包含IMAGE_TOKEN_INDEX
            from utils.utils import IMAGE_TOKEN_INDEX
            has_image_token = (input_ids == IMAGE_TOKEN_INDEX).any()
            print(f"手动检查: input_ids中是否包含IMAGE_TOKEN_INDEX: {has_image_token}")
            
            # 使用模型的_get_image_nums_and_video_nums方法计算image_grid_thw
            image_nums, video_nums = self.model.model._get_image_nums_and_video_nums(input_ids)
            print(f"image_nums: {image_nums}, video_nums: {video_nums}")
            
            # 如果模型方法返回0但手动检查发现有图像token，记录警告但不强制修改
            if image_nums.sum() == 0 and has_image_token:
                print("[WARNING] 模型方法返回0但检测到图像token，这可能是数据预处理问题")
                # 不强制修改，使用模型返回的原始结果
            
            # 使用简单方法：直接调用Qwen2.5-VL的forward，确保image_grid_thw不为None
            print("进行完整的视觉-语言推理...")
            
            # 如果image_grid_thw为None，从input_ids中计算
            if image_grid_thw is None:
                print("⚠️ image_grid_thw为None，从input_ids中计算...")
                image_nums, video_nums = self.model.model._get_image_nums_and_video_nums(input_ids)
                # 计算image_grid_thw
                batch_size = input_ids.shape[0]
                # 假设每个样本有1个图像，使用默认的patch大小
                image_grid_thw = torch.tensor([[1, 56, 56]] * batch_size, dtype=torch.long, device=input_ids.device)
                print(f"✅ 计算的image_grid_thw: {image_grid_thw}")
            
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=image_clip,  # 使用正确的参数名
                image_grid_thw=image_grid_thw,  # 传递image_grid_thw参数
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 检查forward方法是否返回了hidden_states
            if not hasattr(outputs, 'hidden_states'):
                raise RuntimeError(f"forward方法没有返回hidden_states，outputs类型: {type(outputs)}")
            
            # 使用forward方法返回的hidden_states
            output_hidden_states = outputs.hidden_states[-1]  # 使用最后一层的hidden states
            
            print(f"最终output_hidden_states类型: {type(output_hidden_states)}")
            if hasattr(output_hidden_states, 'shape'):
                print(f"最终output_hidden_states形状: {output_hidden_states.shape}")
            
            # 使用generate方法生成新的token
            print("使用generate方法生成回答...")
            
            # 获取[SEG] token的ID
            seg_token_id = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
            print(f"[SEG] token ID: {seg_token_id}")
            
            # 不使用强制生成策略，让模型自然生成[SEG] token
            # 这是真正的LISA方法测试：模型应该学会自然生成[SEG] token
            
            # 直接在evaluate方法内部进行推理，避免无限递归
            print("直接在evaluate方法内部进行推理，避免无限递归")
            
            # 准备推理参数，参考原始infer.py的格式
            with torch.no_grad():
                # 参考原始infer.py的格式：resize_list和original_size_list应该是[(height, width)]的格式
                image_height, image_width = image_clip.shape[-2:]
                resize_list = [(image_height, image_width)]
                original_size_list = [(image_height, image_width)]
                
                # 直接进行推理，不调用self.evaluate()
                print("开始进行文本生成和分割推理...")
                
                # 实现真正的推理逻辑：使用贪心解码生成文本
                print("使用贪心解码生成文本...")
                
                # 初始化生成序列
                generated_ids = input_ids.clone()
                seg_token_id = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
                print(f"🔍 [SEG] token ID: {seg_token_id}")
                
                # 检查模型是否学会了生成[SEG] token
                print("🔍 检查模型是否学会了生成[SEG] token...")
                
                # 贪心解码循环
                for step in range(max_new_tokens):
                    # 前向传播获取logits
                    outputs = self.model.model.forward(
                        input_ids=generated_ids,
                        pixel_values=image_clip,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    
                    # 获取下一个token的logits
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # 贪心选择下一个token
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # 将新token添加到序列
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # 打印生成的token信息
                    if step < 10:  # 只打印前10步
                        print(f"步骤 {step+1}: 生成token {next_token.item()}")
                    
                    # 检查是否生成了[SEG] token
                    if next_token.item() == seg_token_id:
                        print(f"✅ 在第{step+1}步生成了[SEG] token ({seg_token_id})")
                        break
                    
                    # 如果模型没有学会生成[SEG] token，停止生成
                    if step == max_new_tokens - 1:
                        print(f"⚠️ 达到最大生成步数({max_new_tokens})，未生成[SEG] token")
                        break
                    
                    # 检查是否生成了EOS token
                    if tokenizer and next_token.item() == tokenizer.eos_token_id:
                        print(f"✅ 在第{step+1}步生成了EOS token")
                        break
                
                print(f"✅ 贪心解码完成，序列长度: {generated_ids.shape[1]}")
                
                # 获取生成序列的隐藏状态
                print("获取生成序列的隐藏状态...")
                forward_output = self.model.model.forward(
                    input_ids=generated_ids,
                    pixel_values=image_clip,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = forward_output.hidden_states
                print(f"✅ 获取隐藏状态完成，形状: {hidden_states[-1].shape}")
            
            # 构造类似generate输出的格式
            class GeneratedOutputs:
                def __init__(self, sequences, hidden_states):
                    self.sequences = sequences
                    self.hidden_states = hidden_states
            
            generated_outputs = GeneratedOutputs(generated_ids, hidden_states)
            
            # 提取输出，使用与GLOVER_plus.py相同的格式
            output_hidden_states = generated_outputs.hidden_states[-1]
            output_ids = generated_outputs.sequences
            print(f"生成的output_ids形状: {output_ids.shape}")
            print(f"生成的output_ids: {output_ids}")
            print(f"隐藏状态形状: {output_hidden_states.shape}")
            
            # 处理隐藏状态 - 关键修复：需要获取完整序列的隐藏状态
            if not hasattr(self.model, 'text_hidden_fcs') or len(self.model.text_hidden_fcs) == 0:
                raise RuntimeError("模型没有text_hidden_fcs属性或为空")
            
            # 关键修复：我们需要重新前向传播以获取完整序列的隐藏状态
            print("重新前向传播以获取完整序列的隐藏状态...")
            
            # 使用生成的完整序列重新前向传播
            with torch.no_grad():
                # 准备attention_mask（全1，因为都是有效token）
                full_attention_mask = torch.ones_like(output_ids)
                
                # 准备图像输入（与原始输入保持一致）
                if image_clip.shape[0] == 1:
                    images_clip_extend = image_clip.expand(output_ids.shape[0], -1, -1, -1).contiguous()
                else:
                    images_clip_extend = image_clip
                
                # 重新前向传播，获取完整序列的隐藏状态
                full_output = self.model.model.forward(
                    pixel_values=images_clip_extend,
                    attention_mask=full_attention_mask,
                    input_ids=output_ids,
                    image_grid_thw=image_grid_thw,  # 传递image_grid_thw参数
                    output_hidden_states=True,
                )
                
                # 获取最后一层的隐藏状态
                full_hidden_states = full_output.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
                print(f"完整序列隐藏状态形状: {full_hidden_states.shape}")
                
                # 通过MLP投影层处理隐藏状态
                projected_hidden_states = self.model.text_hidden_fcs[0](full_hidden_states)
                print(f"投影后隐藏状态形状: {projected_hidden_states.shape}")
            
            # 使用与GLOVER_plus.py相同的简单检测方式
            # 但使用Qwen中SEG token的正确ID
            seg_token_mask = output_ids == self.seg_token_idx
            print(f"[DEBUG] 使用seg_token_idx: {self.seg_token_idx}")
            print(f"[DEBUG] seg_token_mask中True的数量: {seg_token_mask.sum()}")
            print(f"[DEBUG] output_ids形状: {output_ids.shape}")
            print(f"[DEBUG] projected_hidden_states形状: {projected_hidden_states.shape}")
            print(f"[DEBUG] output_ids内容: {output_ids}")
            print(f"[DEBUG] 查找[SEG] token位置: {(output_ids == self.seg_token_idx).nonzero()}")
            
            # 检查维度匹配
            if seg_token_mask.shape != projected_hidden_states.shape[:2]:
                print(f"⚠️ 维度不匹配: seg_token_mask {seg_token_mask.shape} vs projected_hidden_states {projected_hidden_states.shape[:2]}")
                # 调整seg_token_mask的维度
                if seg_token_mask.shape[1] > projected_hidden_states.shape[1]:
                    seg_token_mask = seg_token_mask[:, :projected_hidden_states.shape[1]]
                elif seg_token_mask.shape[1] < projected_hidden_states.shape[1]:
                    # 在右侧填充False
                    padding = torch.zeros((seg_token_mask.shape[0], projected_hidden_states.shape[1] - seg_token_mask.shape[1])).bool().to(seg_token_mask.device)
                    seg_token_mask = torch.cat([seg_token_mask, padding], dim=1)
            
            # 提取[SEG] token的投影后嵌入
            pred_embeddings = projected_hidden_states[seg_token_mask]
            print(f"提取的[SEG] token嵌入形状: {pred_embeddings.shape}")
            
            # 处理分割token偏移
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat([
                torch.zeros(1).long().to(input_ids.device), seg_token_offset
            ], dim=0)
            
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            print(f"处理后的pred_embeddings数量: {len(pred_embeddings)}")
            if len(pred_embeddings) > 0:
                print(f"第一个pred_embedding形状: {pred_embeddings[0].shape}")
            
            # 获取视觉嵌入
            print("获取SAM视觉嵌入...")
            # 需要为SAM预处理图像，使用与原始infer.py相同的流程
            try:
                # 检查image的类型和格式
                if isinstance(image, torch.Tensor):
                    # 如果image已经是tensor，检查是否需要预处理
                    if image.shape[-1] != 1024 or image.shape[-2] != 1024:
                        print(f"⚠️ 图像尺寸不匹配SAM要求: {image.shape}")
                        print("⚠️ 需要重新预处理图像...")
                        
                        # 重新预处理图像，使用与infer.py相同的流程
                        import torch.nn.functional as F
                        
                        # 将tensor转换回numpy进行预处理
                        if image.dim() == 4:
                            image_np = image[0].permute(1, 2, 0).cpu().numpy()
                        else:
                            image_np = image.permute(1, 2, 0).cpu().numpy()
                        
                        # 使用ResizeLongestSide变换（与infer.py一致）
                        from model.segment_anything.utils.transforms import ResizeLongestSide
                        transform = ResizeLongestSide(1024)
                        image_resized = transform.apply_image(image_np)
                        resize_list = [image_resized.shape[:2]]
                        
                        # 使用preprocess函数标准化和填充（与infer.py一致）
                        def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                                     pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
                                     img_size=1024):
                            x = (x - pixel_mean) / pixel_std
                            h, w = x.shape[-2:]
                            padh = img_size - h
                            padw = img_size - w
                            x = F.pad(x, (0, padw, 0, padh))
                            return x
                        
                        image = preprocess(torch.from_numpy(image_resized).permute(2, 0, 1).contiguous())
                        if image.dim() == 3:
                            image = image.unsqueeze(0)
                        image = image.to(pred_embeddings[0].device)
                        
                        print(f"✅ 重新预处理完成，新尺寸: {image.shape}")
                    else:
                        print(f"✅ 图像尺寸正确: {image.shape}")
                else:
                    print(f"⚠️ 图像类型不是tensor: {type(image)}")
                    # 创建一个空的图像嵌入
                    image_embeddings = torch.zeros((1, 256, 64, 64), device=pred_embeddings[0].device, dtype=pred_embeddings[0].dtype)
                    return pred_masks
                
                # 现在使用正确预处理的图像获取SAM视觉嵌入
                image_embeddings = self.get_visual_embs(image)
                print(f"✅ 成功获取SAM视觉嵌入，形状: {image_embeddings.shape}")
                
            except Exception as e:
                print(f"⚠️ SAM视觉编码失败: {e}")
                print("⚠️ 跳过SAM视觉编码，使用空嵌入")
                # 创建一个空的图像嵌入，形状与SAM期望的匹配
                # SAM期望的形状通常是 [batch_size, 256, 64, 64]
                image_embeddings = torch.zeros((1, 256, 64, 64), device=pred_embeddings[0].device, dtype=pred_embeddings[0].dtype)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model1.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(0).unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model1.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model1.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                if use_text_emb_in_suffix_sam:
                    text_embeds_val = pred_embeddings[i].unsqueeze(0).unsqueeze(1)
                else:
                    text_embeds_val = None

                (
                    sparse_embeddings1,
                    dense_embeddings1,
                ) = self.model.visual_model1.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=low_res_masks,
                    text_embeds=text_embeds_val,
                )
                sparse_embeddings1 = sparse_embeddings1.to(pred_embeddings[i].dtype)
                low_res_masks1, iou_predictions1 = (
                    self.model.visual_model1.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.model.visual_model1.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings1,
                        dense_prompt_embeddings=dense_embeddings1,
                        multimask_output=multimask_output,
                    )
                )
                pred_mask = self.model.visual_model1.postprocess_masks(
                    low_res_masks1,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                print(f"pred_mask形状: {pred_mask.shape}")
                pred_masks.append(pred_mask[:, 0])
            
            return output_ids, pred_masks
        
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        # 确保所有输入都在同一个设备上
        device = images.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # 对于Qwen2.5-VL，mm_projector和vision_tower可能为None
                if (self.get_model().mm_projector is not None and 
                    vision_tower is not None and 
                    hasattr(vision_tower, 'dummy_feature')):
                    cur_input_embeds = (
                        cur_input_embeds
                        + (
                            0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                        ).sum()
                    )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            # 确保所有张量都在同一个设备上
            device = cur_new_input_embeds[0].device
            cur_new_input_embeds = [emb.to(device) for emb in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            
            # 同样对labels进行填充处理
            if labels is not None:
                new_labels_align = []
                for cur_new_labels in new_labels:
                    cur_new_labels = torch.cat(
                        (
                            cur_new_labels,
                            torch.full(
                                (max_len - cur_new_labels.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_labels.dtype,
                                device=cur_new_labels.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_labels)
                new_labels = torch.stack(new_labels_align, dim=0)
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

        # 调试：输出张量形状
        if new_input_embeds is not None and new_labels is not None:
            print(f"[DEBUG] new_input_embeds shape: {new_input_embeds.shape}")
            print(f"[DEBUG] new_labels shape: {new_labels.shape}")
        
        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.kl_loss_weight = kwargs.pop("kl_loss_weight", None)
        else:
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.kl_loss_weight = kwargs.pop("kl_loss_weight", None)
            config.mm_vision_tower = config.vision_tower

        # 添加千问模型缺失的配置属性
        if not hasattr(config, "mm_vision_select_layer"):
            config.mm_vision_select_layer = -2
        if not hasattr(config, "mm_vision_select_feature"):
            config.mm_vision_select_feature = "patch"
        if not hasattr(config, "pretrain_mm_mlp_adapter"):
            config.pretrain_mm_mlp_adapter = None
        if not hasattr(config, "mm_use_im_start_end"):
            config.mm_use_im_start_end = True
        
        # 存储精度信息
        self.precision = kwargs.get("precision", "fp32")

        self.seg_token_idx = kwargs.pop("seg_token_idx", None)  # 必须显式传递seg_token_idx
        if self.seg_token_idx is None:
            raise ValueError("seg_token_idx must be provided when initializing GloverQwenForCausalLM")

        PreTrainedModel.__init__(self, config)

        self.model = GloverQwenModel(config, **kwargs)
        
        # 在设置precision后，重新设置SAM模型的精度
        if hasattr(self.model, '_setup_sam_precision_and_device'):
            self.model._setup_sam_precision_and_device(config)
        
        # 添加lm_head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 千问模型特有的初始化
        self.model.initialize_glover_modules(self.model.config)
        
        # 初始化Qwen2.5-VL的processor
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model.config.name_or_path,
                trust_remote_code=True
            )
            print(f"✅ 成功初始化Qwen2.5-VL processor: {self.processor}")
        except Exception as e:
            print(f"⚠️ 无法初始化processor: {e}")
            self.processor = None
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """自定义from_pretrained方法"""
        from transformers import AutoConfig
        
        try:
            # 尝试加载Qwen2.5-VL配置
            from transformers import Qwen2_5_VLConfig
            config = Qwen2_5_VLConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        except Exception as e:
            print(f"无法加载Qwen2.5-VL配置: {e}")
            try:
                # 回退到Qwen2-VL配置
                from transformers import Qwen2VLConfig
                config = Qwen2VLConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            except Exception as e2:
                print(f"无法加载Qwen2-VL配置: {e2}")
                try:
                    # 回退到Qwen2配置
                    from transformers import Qwen2Config
                    config = Qwen2Config.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
                except Exception as e3:
                    print(f"无法加载Qwen2配置: {e3}")
                    print(f"[ERROR] 无法加载任何Qwen配置，这可能导致模型结构不匹配。")
                    print(f"[ERROR] 请检查模型路径 {pretrained_model_name_or_path} 是否正确，或确保模型文件完整。")
                    raise RuntimeError(f"无法加载任何Qwen配置。请检查模型路径 {pretrained_model_name_or_path} 是否正确。错误: {e3}")
        
        # 设置配置的模型路径
        config._name_or_path = pretrained_model_name_or_path
        
        # 创建模型实例
        model = cls(config, **kwargs)
        
        try:
            # 尝试加载Qwen2.5-VL模型
            from transformers import Qwen2_5_VLForConditionalGeneration
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            model.model.model = qwen_model  # 加载到GloverQwenModel内部的model属性
            
            # 使用Qwen模型原有的lm_head，而不是创建新的
            model.lm_head = qwen_model.lm_head
            
            # 初始化processor
            model.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
            print("✅ 成功加载 Qwen2.5-VL 预训练模型和processor")
        except Exception as e:
            print(f"加载Qwen2.5-VL模型失败: {e}")

        
        return model
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(self.model, 'enable_input_require_grads'):
            self.model.enable_input_require_grads()
            
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    def get_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.model.set_input_embeddings(value)

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, 
                inputs_embeds=None, labels=None, use_cache=None, 
                output_attentions=None, output_hidden_states=None, 
                return_dict=None, **kwargs):
        """前向传播方法，支持文本生成和多模态处理"""
        
        # 确保GLOVER模块已初始化
        if hasattr(self.model, '_ensure_glover_initialized'):
            self.model._ensure_glover_initialized()
        
        # 如果有inputs_embeds，说明是多模态输入，直接使用底层的Qwen模型
        if inputs_embeds is not None:
            return self.model.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # 如果是生成阶段（有past_key_values），使用底层Qwen模型
        if past_key_values is not None:
            return self.model.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        # 否则使用训练时的model_forward
        # 确保传递所有必需的参数
        model_kwargs = {}
        
        # 调试：打印kwargs的键
        print(f"🔍 model forward kwargs 包含的键: {list(kwargs.keys())}")
        
        # 第一步：从函数参数中获取input_ids、labels、attention_mask
        # PEFT将input_ids、labels、attention_mask作为函数参数传递，而不是kwargs
        if input_ids is not None:
            model_kwargs['input_ids'] = input_ids
            print(f"✅ 从函数参数中获取 input_ids: {input_ids.shape}")
        if labels is not None:
            model_kwargs['labels'] = labels
            print(f"✅ 从函数参数中获取 labels: {labels.shape}")
        if attention_mask is not None:
            model_kwargs['attention_mask'] = attention_mask
            print(f"✅ 从函数参数中获取 attention_mask: {attention_mask.shape}")
        
        # 第二步：从kwargs中获取其他参数
        # 检查是否使用了processor处理的数据
        if 'pixel_values' in kwargs and 'image_grid_thw' in kwargs:
            print(f"✅ 检测到processor处理的数据")
            model_kwargs['pixel_values'] = kwargs['pixel_values']
            model_kwargs['image_grid_thw'] = kwargs['image_grid_thw']
            # 对于processor模式，仍然需要原始图像数据用于SAM处理
            kwargs_args = ['images', 'offset', 'masks_list', 'resize_list']
        else:
            print(f"✅ 使用传统模式数据")
            kwargs_args = ['images', 'images_clip', 'offset', 'masks_list', 'resize_list']
        
        for arg in kwargs_args:
            if arg in kwargs:
                model_kwargs[arg] = kwargs[arg]
                print(f"✅ 从kwargs中找到参数 {arg}")
            else:
                print(f"❌ 缺少参数 {arg}")
        
        # 第三步：最终检查，确保所有必需参数都存在
        if 'pixel_values' in model_kwargs:
            # processor模式
            required_args = ['input_ids', 'labels', 'attention_mask', 'pixel_values', 'image_grid_thw', 'offset', 'masks_list', 'resize_list']
        else:
            # 传统模式
            required_args = ['input_ids', 'labels', 'attention_mask', 'images', 'images_clip', 'offset', 'masks_list', 'resize_list']
        
        missing_args = [arg for arg in required_args if arg not in model_kwargs]
        if missing_args:
            # 检查是否启用了strict模式
            strict_mode = kwargs.get('strict_mode', False)
            if strict_mode:
                # Strict模式：直接抛出异常
                raise ValueError(f"Strict模式：模型forward缺少必需参数: {missing_args}。请检查数据管道配置。")
            else:
                # 宽松模式：打印警告
                print(f"⚠️ 最终检查：仍然缺少参数: {missing_args}")
                print(f"   建议在训练脚本中启用 --strict_mode 来检测数据管道问题")
        
        # 添加可选参数
        model_kwargs['inference'] = kwargs.get('inference', False)
        model_kwargs['use_text_emb_in_suffix_sam'] = kwargs.get('use_text_emb_in_suffix_sam', False)
        model_kwargs['tokenizer'] = kwargs.get('tokenizer', None)
        model_kwargs['strict_mode'] = kwargs.get('strict_mode', False)
        
        return self.model_forward(**model_kwargs)
    
    def compute_segmentation_loss(self, pred_masks, gt_masks, inference=False):
        """
        计算分割损失：与GLOVER Plus一致的损失计算方式
        使用sigmoid_focal_loss + KL损失
        """
        if pred_masks is None or gt_masks is None:
            return torch.tensor(0.0, device=pred_masks.device if pred_masks is not None else gt_masks.device), torch.tensor(0.0, device=gt_masks.device if gt_masks is not None else pred_masks.device)
        
        # 确保数据类型一致（损失计算通常使用float32以获得更好的数值稳定性）
        pred_masks = pred_masks.float()
        gt_masks = gt_masks.float()
        
        # 按照GLOVER Plus的方式计算损失
        mask_focal_loss = 0
        num_masks = 0
        kl_loss = 0
        
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            
            # 确保形状匹配
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            
            # 计算sigmoid_focal_loss
            mask_focal_loss += (
                sigmoid_focal_loss(pred_mask, gt_mask, num_boxes=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
            
            # 计算KL损失
            kl_loss += cal_kl(pred_mask, gt_mask) * gt_mask.shape[0]
        
        # 按照GLOVER Plus的方式归一化损失
        mask_loss = mask_focal_loss * 0.1 / (num_masks + 1e-8)
        kl_loss = kl_loss / (num_masks + 1e-8)
        kl_loss = kl_loss * self.kl_loss_weight
        
        return mask_loss, kl_loss

    def model_forward(
        self,
        images: torch.FloatTensor = None,
        images_clip: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
        offset: torch.LongTensor = None,
        masks_list: List[torch.FloatTensor] = None,
        resize_list: List[tuple] = None,
        inference: bool = False,
        use_text_emb_in_suffix_sam: bool = False,
        # 新增：processor处理后的数据
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        **kwargs,
    ):
        # 检查是否使用了processor处理的数据
        if pixel_values is not None and image_grid_thw is not None:
            print(f"[DEBUG] 使用processor处理的数据，直接调用模型")
            print(f"[DEBUG] pixel_values形状: {pixel_values.shape}")
            print(f"[DEBUG] input_ids形状: {input_ids.shape}")
            print(f"[DEBUG] image_grid_thw形状: {image_grid_thw.shape}")
            
            # 直接使用processor处理后的数据调用模型
            output = self.model.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                output_hidden_states=True,
            )
            
            print(f"[DEBUG] 模型输出形状: {output.logits.shape}")
            print(f"[DEBUG] 模型输出键: {list(output.keys())}")
            
            # 继续执行GLOVER的分割训练逻辑，而不是直接返回
            print(f"[DEBUG] 继续执行GLOVER分割训练逻辑...")
            
            # 设置变量以匹配processor处理后的数据
            new_input_ids = input_ids
            new_attention_mask = attention_mask
            new_labels = labels
            
            # 获取隐藏状态用于分割训练
            output_hidden_states = output.hidden_states
            
            # 调试：检查模型输出的维度
            print(f"[DEBUG] 模型输出分析:")
            print(f"  output类型: {type(output)}")
            print(f"  output.logits形状: {output.logits.shape if hasattr(output, 'logits') else '无logits'}")
            print(f"  output.hidden_states长度: {len(output_hidden_states) if output_hidden_states else '无hidden_states'}")
            if output_hidden_states:
                print(f"  output_hidden_states[-1]形状: {output_hidden_states[-1].shape}")
            print(f"  input_ids形状: {input_ids.shape}")
            print(f"  attention_mask形状: {attention_mask.shape}")
            print(f"  image_grid_thw: {image_grid_thw}")
            
            # LISA方法：通过MLP投影层处理隐藏状态 (γ: h̃_seg → h_seg)
            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            
            # 调试：检查MLP投影层的输入输出维度
            print(f"[DEBUG] MLP投影层分析:")
            print(f"  text_hidden_fcs[0]输入维度: {self.model.text_hidden_fcs[0][0].in_features}")
            print(f"  text_hidden_fcs[0]输出维度: {self.model.text_hidden_fcs[0][2].out_features}")
            print(f"  output_hidden_states[-1]形状: {output_hidden_states[-1].shape}")
            
            # 这里的text_hidden_fcs[0]就是LISA论文中的MLP投影层γ
            # 确保text_hidden_fcs在正确的设备和数据类型上
            device = output_hidden_states[-1].device
            dtype = output_hidden_states[-1].dtype
            self.model.text_hidden_fcs = self.model.text_hidden_fcs.to(device=device, dtype=dtype)
            projected_hidden = self.model.text_hidden_fcs[0](output_hidden_states[-1])
            print(f"  MLP投影后形状: {projected_hidden.shape}")
            
            hidden_states.append(projected_hidden)
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            print(f"[DEBUG] LISA - MLP投影后的隐藏状态 h_seg: {last_hidden_state.shape}")

            # LISA方法：提取[SEG] token的最后一层隐藏嵌入 (h̃_seg)
            print(f"[DEBUG] 使用processor输出的token序列:")
            print(f"  last_hidden_state形状: {last_hidden_state.shape}")
            print(f"  processor处理后的input_ids形状: {new_input_ids.shape}")
            print(f"  seg_token_idx: {self.seg_token_idx}")
            
            # 直接使用processor处理后的input_ids创建seg_token_mask
            # 这确保了seg_token_mask与last_hidden_state完全对齐
            seg_token_mask = new_input_ids == self.seg_token_idx
            print(f"  [DEBUG] 使用processor输出的input_ids创建seg_token_mask")
            print(f"  [DEBUG] seg_token_mask形状: {seg_token_mask.shape}")
            print(f"  [DEBUG] seg_token_mask中True的数量: {seg_token_mask.sum()}")
            
            # 验证维度对齐
            if seg_token_mask.shape != last_hidden_state.shape[:2]:
                print(f"  [ERROR] 维度不匹配: seg_token_mask {seg_token_mask.shape} vs last_hidden_state {last_hidden_state.shape[:2]}")
                print(f"  [WARNING] 跳过这个样本，不创建假掩码")
                return {
                    "loss": torch.tensor(0.0, device=last_hidden_state.device, dtype=last_hidden_state.dtype),
                    "ce_loss": torch.tensor(0.0, device=last_hidden_state.device, dtype=last_hidden_state.dtype),
                    "mask_loss": torch.tensor(0.0, device=last_hidden_state.device, dtype=last_hidden_state.dtype),
                    "kl_loss": torch.tensor(0.0, device=last_hidden_state.device, dtype=last_hidden_state.dtype),
                    "logits": output.logits,
                    "hidden_states": output.hidden_states,
                    "rope_deltas": getattr(output, 'rope_deltas', None),
                    "pred_masks": [],
                }
            else:
                print(f"  [DEBUG] 维度对齐成功")
            
            print(f"  最终seg_token_mask形状: {seg_token_mask.shape}")
            print(f"  最终seg_token_mask中True的数量: {seg_token_mask.sum()}")
            
            # 提取[SEG] token的隐藏状态
            seg_token_embeddings = last_hidden_state[seg_token_mask]
            print(f"[DEBUG] 提取的[SEG] token嵌入形状: {seg_token_embeddings.shape}")
            
            if seg_token_embeddings.shape[0] == 0:
                # 训练时：即使没有[SEG] token也要进行文本生成训练
                # 推理时：没有[SEG] token就跳过分割
                print(f"[ERROR] 没有找到[SEG] token！")
                print(f"[ERROR] seg_token_idx: {self.seg_token_idx}")
                print(f"[ERROR] input_ids形状: {new_input_ids.shape}")
                print(f"[ERROR] input_ids内容: {new_input_ids}")
                print(f"[ERROR] seg_token_mask形状: {seg_token_mask.shape}")
                print(f"[ERROR] seg_token_mask内容: {seg_token_mask}")
                print(f"[ERROR] seg_token_mask中True的数量: {seg_token_mask.sum()}")
                
                # 检查input_ids中是否包含[SEG] token
                seg_token_found = (new_input_ids == self.seg_token_idx).any()
                print(f"[ERROR] input_ids中是否包含[SEG] token: {seg_token_found}")
                
                if inference:
                    print(f"[WARNING] 推理模式：没有找到[SEG] token，跳过分割训练")
                    # 返回只有语言模型损失的结果，但确保包含所有损失字段
                    # 推理时output.loss可能为None，需要处理
                    device = last_hidden_state.device
                    dtype = last_hidden_state.dtype
                    return {
                        "loss": torch.tensor(0.0, device=device, dtype=dtype),
                        "ce_loss": torch.tensor(0.0, device=device, dtype=dtype),
                        "mask_loss": torch.tensor(0.0, device=device, dtype=dtype),
                        "kl_loss": torch.tensor(0.0, device=device, dtype=dtype),
                        "logits": output.logits,
                        "hidden_states": output.hidden_states,
                        "rope_deltas": getattr(output, 'rope_deltas', None),
                    }
                else:
                    print(f"[INFO] 训练模式：没有找到[SEG] token，只进行文本生成训练")
                    # 训练时：即使没有[SEG] token也要返回语言模型损失，让模型学会生成[SEG] token
                    device = last_hidden_state.device
                    dtype = last_hidden_state.dtype
                    return {
                        "loss": output.loss if output.loss is not None else torch.tensor(0.0, device=device, dtype=dtype),
                        "ce_loss": output.loss if output.loss is not None else torch.tensor(0.0, device=device, dtype=dtype),
                        "mask_loss": torch.tensor(0.0, device=device, dtype=dtype),
                        "kl_loss": torch.tensor(0.0, device=device, dtype=dtype),
                        "logits": output.logits,
                        "hidden_states": output.hidden_states,
                        "rope_deltas": getattr(output, 'rope_deltas', None),
                        "pred_masks": [],
                    }
            
            # 使用SAM的mask_decoder生成掩码预测（按照原始GLOVER的方式）
            print(f"[DEBUG] 使用SAM mask_decoder生成掩码预测")
            
            # 确保visual_model1在正确的设备上（精度已在初始化时设置）
            if hasattr(self.model, 'visual_model1'):
                visual_model1 = self.model.visual_model1
            else:
                print(f"[ERROR] 没有找到visual_model1（SAM模型）")
                # 返回只有语言模型损失的结果，但确保包含所有损失字段
                return {
                    "loss": output.loss,
                    "ce_loss": output.loss,
                    "mask_loss": torch.tensor(0.0, device=output.loss.device, dtype=output.loss.dtype),
                    "kl_loss": torch.tensor(0.0, device=output.loss.device, dtype=output.loss.dtype),
                    "logits": output.logits,
                    "hidden_states": output.hidden_states,
                    "rope_deltas": getattr(output, 'rope_deltas', None),
                }
            
            # 按照原始GLOVER的方式处理每个样本
            pred_masks = []
            
            # 获取所有相关组件的设备信息
            prompt_encoder_device = next(visual_model1.prompt_encoder.parameters()).device
            mask_decoder_device = next(visual_model1.mask_decoder.parameters()).device
            seg_token_device = seg_token_embeddings.device
            pixel_values_device = pixel_values.device
            
            print(f"[DEBUG] 设备信息: prompt_encoder={prompt_encoder_device}, mask_decoder={mask_decoder_device}, seg_token={seg_token_device}, pixel_values={pixel_values_device}")
            
            for i in range(len(seg_token_embeddings)):
                # 获取当前样本的seg_token_embedding
                seg_embedding = seg_token_embeddings[i]  # [256]
                print(f"[DEBUG] seg_embedding原始形状: {seg_embedding.shape}")
                
                # 确保seg_embedding在正确的设备上
                seg_embedding = seg_embedding.to(prompt_encoder_device)
                
                # 使用SAM的prompt_encoder处理文本嵌入
                # seg_embedding是[256]，需要变成3维张量[1, 1, 256]给SAM的prompt_encoder
                text_embeds = seg_embedding.unsqueeze(0).unsqueeze(1)  # [256] -> [1, 1, 256]
                print(f"[DEBUG] text_embeds形状: {text_embeds.shape}")
                
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = visual_model1.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=text_embeds,
                )
                
                # 获取图像嵌入（从pixel_values中提取）
                # 这里需要根据image_grid_thw来确定每个样本对应的图像特征
                if i < image_grid_thw.shape[0]:
                    # 计算当前样本的图像特征范围
                    start_idx = sum(image_grid_thw[:i, 0].sum().item() for _ in range(i)) if i > 0 else 0
                    end_idx = start_idx + image_grid_thw[i, 0].item() * image_grid_thw[i, 1].item() * image_grid_thw[i, 2].item()
                    
                    # 关键修复：使用原始图像数据通过SAM的image_encoder处理
                    print(f"[DEBUG] 使用原始图像数据通过SAM的image_encoder处理")
                    
                    # 从原始图像数据中获取当前样本的图像
                    if images is not None:
                        sample_image = images[i].unsqueeze(0)  # [1, C, H, W]
                        sample_image = sample_image.to(mask_decoder_device)
                        
                        print(f"[DEBUG] 原始图像形状: {sample_image.shape}")
                        
                        # 使用SAM的image_encoder处理图像
                        with torch.no_grad():
                            # 确保图像数据类型与SAM模型匹配
                            # 获取SAM模型的权重数据类型
                            sam_weight_dtype = next(visual_model1.image_encoder.parameters()).dtype
                            print(f"[DEBUG] SAM模型权重数据类型: {sam_weight_dtype}")
                            print(f"[DEBUG] 输入图像数据类型: {sample_image.dtype}")
                            
                            # 将输入图像转换为与SAM模型权重相同的数据类型
                            sample_image = sample_image.to(dtype=sam_weight_dtype)
                            print(f"[DEBUG] 转换后图像数据类型: {sample_image.dtype}")
                            
                            # 检查图像尺寸是否符合SAM的要求
                            if sample_image.shape[2:] != (1024, 1024):
                                print(f"[WARNING] 图像尺寸{sample_image.shape[2:]}不符合SAM要求(1024,1024)，跳过这个样本")
                                continue
                            
                            print(f"[DEBUG] 图像尺寸符合要求: {sample_image.shape}")
                            sample_image_embeddings = visual_model1.image_encoder(sample_image)
                        
                        print(f"[DEBUG] SAM image_encoder输出形状: {sample_image_embeddings.shape}")
                    else:
                        print(f"[ERROR] 没有找到原始图像数据，无法进行分割预测")
                        # 如果没有原始图像数据，跳过这个样本
                        continue
                    
                    # 确保所有张量在正确的设备和数据类型上
                    # 使用mask_decoder的实际权重数据类型
                    target_dtype = next(visual_model1.mask_decoder.parameters()).dtype
                    print(f"[DEBUG] mask_decoder权重数据类型: {target_dtype}")
                    
                    sparse_embeddings = sparse_embeddings.to(mask_decoder_device).to(target_dtype)
                    dense_embeddings = dense_embeddings.to(mask_decoder_device).to(target_dtype)
                    
                    # 确保image_embeddings也是正确的数据类型
                    if sample_image_embeddings.dtype != target_dtype:
                        sample_image_embeddings = sample_image_embeddings.to(target_dtype)
                    
                    # 确保image_pe也是正确的数据类型
                    image_pe = visual_model1.prompt_encoder.get_dense_pe().to(mask_decoder_device).to(target_dtype)
                    
                    # 使用SAM的mask_decoder生成掩码
                    multimask_output = False
                    low_res_masks, iou_predictions = visual_model1.mask_decoder(
                        image_embeddings=sample_image_embeddings,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    
                    # 后处理掩码
                    if masks_list is not None and i < len(masks_list) and masks_list[i] is not None:
                        # 确保masks_list在正确的设备和数据类型上
                        mask = masks_list[i].to(mask_decoder_device).to(target_dtype)
                        pred_mask = visual_model1.postprocess_masks(
                            low_res_masks,
                            input_size=resize_list[i] if resize_list and i < len(resize_list) else (1024, 1024),
                            original_size=mask.shape[1:],
                        )
                        pred_masks.append(pred_mask[:, 0])
                    else:
                        # 如果没有真实掩码，使用默认尺寸
                        pred_mask = visual_model1.postprocess_masks(
                            low_res_masks,
                            input_size=(1024, 1024),
                            original_size=(1024, 1024),
                        )
                        pred_masks.append(pred_mask[:, 0])
                else:
                    # 没有原始图像数据，跳过这个样本，不创建假掩码
                    print(f"[DEBUG] 样本{i}没有原始图像数据，跳过")
                    continue
            
            print(f"[DEBUG] 生成的掩码数量: {len(pred_masks)}")
            if len(pred_masks) > 0:
                print(f"[DEBUG] 第一个掩码形状: {pred_masks[0].shape}")
            
            # 计算分割损失
            if masks_list is not None and len(masks_list) > 0 and len(pred_masks) > 0:
                # 处理真实掩码
                gt_masks = []
                # 使用mask_decoder的实际权重数据类型（与前面保持一致）
                target_dtype = next(visual_model1.mask_decoder.parameters()).dtype
                print(f"[DEBUG] 损失计算使用数据类型: {target_dtype}")
                
                for i, mask in enumerate(masks_list):
                    if mask is not None and i < len(pred_masks):
                        # 确保mask在正确的设备和数据类型上
                        mask = mask.to(mask_decoder_device).to(target_dtype)
                        
                        # 检查掩码尺寸是否匹配，不匹配则跳过
                        if mask.shape != pred_masks[i].shape:
                            print(f"[WARNING] 掩码尺寸不匹配，跳过: gt={mask.shape}, pred={pred_masks[i].shape}")
                            continue
                        gt_masks.append(mask)
                    elif i < len(pred_masks):
                        # 没有真实掩码，跳过这个样本，不创建假掩码
                        print(f"[DEBUG] 样本{i}没有真实掩码，跳过")
                        continue
                
                if len(gt_masks) > 0:
                    # 按照GLOVER_plus.py的方式处理掩码，不强行对齐尺寸
                    mask_focal_loss = 0
                    num_masks = 0
                    kl_loss = 0
                    
                    for batch_idx in range(len(pred_masks)):
                        if batch_idx >= len(gt_masks) or gt_masks[batch_idx] is None:
                            continue
                            
                        pred_mask = pred_masks[batch_idx]
                        gt_mask = gt_masks[batch_idx]
                        
                        if pred_mask is None:
                            continue
                        
                        # 确保设备一致
                        pred_mask = pred_mask.to(mask_decoder_device).to(target_dtype)
                        gt_mask = gt_mask.to(mask_decoder_device).to(target_dtype)
                        
                        # 按照GLOVER_plus.py的方式处理维度
                        if len(gt_mask.shape) == 2 and len(pred_mask.shape) == 3:
                            # gt_mask是[H, W]，pred_mask是[N, H, W]，取pred_mask的第一个
                            pred_mask = pred_mask[0]
                        elif len(gt_mask.shape) == 3 and len(pred_mask.shape) == 2:
                            # gt_mask是[N, H, W]，pred_mask是[H, W]，取gt_mask的第一个
                            gt_mask = gt_mask[0]
                        elif len(gt_mask.shape) == 3 and len(pred_mask.shape) == 3:
                            # 都是3维，检查第一个维度是否匹配
                            if gt_mask.shape[0] != pred_mask.shape[0]:
                                print(f"[WARNING] 掩码batch维度不匹配: gt={gt_mask.shape}, pred={pred_mask.shape}")
                                continue
                            # 取第一个掩码
                            gt_mask = gt_mask[0]
                            pred_mask = pred_mask[0]
                        
                        # 检查最终形状是否匹配（与GLOVER_plus.py一致）
                        if gt_mask.shape != pred_mask.shape:
                            print(f"[WARNING] 掩码形状不匹配，跳过: gt={gt_mask.shape}, pred={pred_mask.shape}")
                            continue
                        
                        # 计算损失（与GLOVER_plus.py一致）
                        mask_focal_loss += (
                            sigmoid_focal_loss(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0), num_boxes=1)
                            * 1
                        )
                        num_masks += 1
                        kl_loss += cal_kl(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0)) * 1
                    
                    if num_masks > 0:
                        # 按照GLOVER_plus.py的方式归一化损失
                        mask_loss = mask_focal_loss * 0.1 / (num_masks + 1e-8)
                        kl_loss = kl_loss / (num_masks + 1e-8)
                        kl_loss = kl_loss * self.kl_loss_weight
                        
                        print(f"[DEBUG] 分割损失: mask_loss={mask_loss.item():.6f}, kl_loss={kl_loss.item():.6f}")
                        
                        # 将分割损失添加到总损失中
                        total_loss = output.loss + mask_loss + kl_loss
                        output.loss = total_loss
                        print(f"[DEBUG] 总损失: {total_loss.item():.6f}")
                    else:
                        print(f"[WARNING] 没有有效的掩码对，跳过分割损失计算")
                        mask_loss = torch.tensor(0.0, device=output.loss.device, dtype=output.loss.dtype)
                        kl_loss = torch.tensor(0.0, device=output.loss.device, dtype=output.loss.dtype)
                else:
                    print(f"[WARNING] 没有有效的真实掩码，跳过分割损失计算")
            else:
                print(f"[WARNING] masks_list为空，跳过分割损失计算")
            
            # 返回包含分割损失的结果
            # 推理时output.loss可能为None，需要处理
            if output.loss is not None:
                device = output.loss.device
                dtype = output.loss.dtype
            else:
                # 推理时使用last_hidden_state的设备类型
                device = last_hidden_state.device
                dtype = last_hidden_state.dtype
            
            return {
                "loss": output.loss if output.loss is not None else torch.tensor(0.0, device=device, dtype=dtype),
                "ce_loss": output.loss if output.loss is not None else torch.tensor(0.0, device=device, dtype=dtype),
                "mask_loss": mask_loss if 'mask_loss' in locals() else torch.tensor(0.0, device=device, dtype=dtype),
                "kl_loss": kl_loss if 'kl_loss' in locals() else torch.tensor(0.0, device=device, dtype=dtype),
                "logits": output.logits,
                "hidden_states": output.hidden_states,
                "rope_deltas": getattr(output, 'rope_deltas', None),
                "pred_masks": pred_masks if 'pred_masks' in locals() else [],  # 添加生成的掩码
            }
            
        else:
            # 传统模式：使用images和images_clip
            print(f"[DEBUG] 使用传统模式处理数据")
            
            # 获取图像嵌入
            image_embeddings = self.get_visual_embs(images)
            batch_size = image_embeddings.shape[0]
            assert batch_size == len(offset) - 1
            
            # 调试：检查输入图像的特征
            print(f"[DEBUG] images形状: {images.shape}")
            print(f"[DEBUG] images_clip形状: {images_clip.shape}")
            print(f"[DEBUG] image_embeddings形状: {image_embeddings.shape}")

            # 处理segmentation token - 使用与GLOVER_plus.py相同的简单方式
            # 注意：这里使用原始的input_ids来生成seg_token_mask，因为我们需要找到[SEG] token的位置
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    seg_token_mask,
                    torch.zeros((seg_token_mask.shape[0], 1)).bool().to(input_ids.device),
                ],
                dim=1,
            )
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 255)).bool().to(input_ids.device), seg_token_mask],
                dim=1,
            )

        if inference:
            # 推理模式 - 使用CLIP特征，像原始GLOVER一样
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                # 使用简单方法：直接调用Qwen2.5-VL的forward，不传递image_grid_thw
                inputs_i = {
                    'pixel_values': images_clip_extend[: end_i - start_i],
                    'attention_mask': attention_mask[start_i:end_i],
                    'input_ids': input_ids[start_i:end_i],
                    'output_hidden_states': True,
                    'return_dict': True,
                }
                output_i = self.model.model.forward(**inputs_i)
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
        else:
            # 训练模式 - 使用CLIP特征，像原始GLOVER一样
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            
            images_clip = torch.cat(images_clip_list, dim=0)

            # 使用正确的参数名pixel_values，但使用images_clip（224x224）而不是images（1024x1024）
            print(f"[DEBUG] 使用CLIP预处理后的图像，参数名：pixel_values")
            print(f"[DEBUG] images_clip实际形状: {images_clip.shape}")
            
            # Qwen2.5-VL需要image_grid_thw参数
            batch_size = images_clip.shape[0]
            # 根据实际图像尺寸设置image_grid_thw
            # images_clip形状应该是 [batch_size, channels, height, width]
            actual_height = images_clip.shape[2]
            actual_width = images_clip.shape[3]
            print(f"[DEBUG] 实际图像尺寸: {actual_height}x{actual_width}")
            
            # 根据测试结果：Qwen2.5-VL使用14x14的patch大小
            # image_grid_thw = [t, h_patches, w_patches]
            # t=1 (静态图像), h_patches=ceil(height/14), w_patches=ceil(width/14)
            import math
            h_patches = math.ceil(actual_height / 14)
            w_patches = math.ceil(actual_width / 14)
            image_grid_thw = torch.tensor([[1, h_patches, w_patches]] * batch_size, dtype=torch.long, device=images_clip.device)
            print(f"[DEBUG] 计算的image_grid_thw: {image_grid_thw}")
            
            # 使用processor处理图像和文本，确保input_ids和labels都经过正确处理
            print(f"[DEBUG] 使用processor处理图像和文本")
            print(f"[DEBUG] images_clip形状: {images_clip.shape}")
            
            if hasattr(self, 'processor') and self.processor is not None:
                # 直接使用processor处理images_clip，不需要转换为base64
                from PIL import Image
                
                # 将tensor转换为PIL图像列表
                pil_images = []
                for i in range(images_clip.shape[0]):
                    # 将tensor转换为PIL图像
                    img_tensor = images_clip[i]
                    # 反归一化并转换为0-255范围
                    img_tensor = (img_tensor + 1) * 127.5  # 假设是[-1,1]范围
                    img_tensor = torch.clamp(img_tensor, 0, 255).byte()
                    img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()
                    
                    # 转换为PIL图像
                    img_pil = Image.fromarray(img_tensor)
                    pil_images.append(img_pil)
                
                # 构造messages格式，包含图像和文本
                messages = []
                target_texts = []  # 用于存储目标文本
                for i in range(input_ids.shape[0]):
                    if 'tokenizer' in kwargs and kwargs['tokenizer'] is not None:
                        text = kwargs['tokenizer'].decode(input_ids[i], skip_special_tokens=False)
                        print(f"[DEBUG] 原始文本 {i}: {text}")
                        print(f"[DEBUG] 原始input_ids {i} 包含[SEG] token: {self.seg_token_idx in input_ids[i]}")
                        
                        # 如果有labels，解码目标文本
                        if labels is not None:
                            # 过滤掉-100值（用于忽略损失）
                            valid_labels = labels[i][labels[i] != -100]
                            if len(valid_labels) > 0:
                                target_text = kwargs['tokenizer'].decode(valid_labels, skip_special_tokens=False)
                                target_texts.append(target_text)
                                print(f"[DEBUG] 目标文本 {i}: {target_text}")
                            else:
                                target_texts.append("")  # 空目标文本
                                print(f"[DEBUG] 目标文本 {i}: 空（所有labels都是-100）")
                        else:
                            target_texts.append("")  # 空目标文本
                            print(f"[DEBUG] 没有目标文本labels")

                    
                    message = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_images[i]},
                            {"type": "text", "text": text}
                        ]
                    }
                    messages.append(message)
                
                # 为每个样本单独处理，避免批次维度问题
                from qwen_vl_utils import process_vision_info
                
                # 处理每个样本
                all_input_ids = []
                all_attention_masks = []
                all_pixel_values = []
                all_image_grid_thw = []
                all_labels = []
                
                for i, message in enumerate(messages):
                    # 为单个样本处理
                    single_messages = [message]
                    image_inputs, video_inputs = process_vision_info(single_messages)
                    
                    # 使用processor处理单个样本
                    text = self.processor.apply_chat_template(single_messages, tokenize=False, add_generation_prompt=True)
                    
                    # 处理单个样本，使用方案A：保持HF原生方式，确保labels和input_ids长度一致
                    if target_texts[i]:  # 如果有目标文本
                        single_inputs = self.processor(
                            text=[text],
                            text_target=[target_texts[i]],  # 使用text_target参数
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        
                        # 确保labels长度和input_ids一致
                        if hasattr(single_inputs, 'labels') and single_inputs.labels is not None:
                            original_labels_len = single_inputs.labels.shape[1]
                            original_input_ids_len = single_inputs.input_ids.shape[1]
                            
                            # 检查原始labels中有效token数量（非-100的token）
                            valid_tokens_before = (single_inputs.labels != -100).sum().item()
                            print(f"[DEBUG] 原始labels中有效token数: {valid_tokens_before}")
                            
                            if original_labels_len != original_input_ids_len:
                                # 计算需要padding的长度
                                pad_len = original_input_ids_len - original_labels_len
                                # 在右侧padding，用-100填充（忽略损失）
                                single_inputs.labels = F.pad(single_inputs.labels, (0, pad_len), value=-100)
                                print(f"[DEBUG] 调整labels长度: {original_labels_len} -> {single_inputs.labels.shape[1]}, padding长度: {pad_len}")
                                
                                # 检查padding后有效token数量是否保持不变
                                valid_tokens_after = (single_inputs.labels != -100).sum().item()
                                print(f"[DEBUG] padding后labels中有效token数: {valid_tokens_after}")
                                
                                if valid_tokens_before != valid_tokens_after:
                                    print(f"[WARNING] 有效token数量发生变化！padding前: {valid_tokens_before}, padding后: {valid_tokens_after}")
                                else:
                                    print(f"[DEBUG] ✅ 有效token数量保持不变，padding成功")
                            else:
                                print(f"[DEBUG] labels长度已匹配input_ids: {original_labels_len}")
                            
                            all_labels.append(single_inputs.labels)
                            print(f"[DEBUG] 最终labels形状: {single_inputs.labels.shape}")

                    all_input_ids.append(single_inputs.input_ids)
                    all_attention_masks.append(single_inputs.attention_mask)
                    all_pixel_values.append(single_inputs.pixel_values)
                    all_image_grid_thw.append(single_inputs.image_grid_thw)
                
                # 合并所有样本
                inputs = type('Inputs', (), {})()
                inputs.input_ids = torch.cat(all_input_ids, dim=0)
                inputs.attention_mask = torch.cat(all_attention_masks, dim=0)
                inputs.pixel_values = torch.cat(all_pixel_values, dim=0)
                inputs.image_grid_thw = torch.cat(all_image_grid_thw, dim=0)
                
                # 合并所有labels（现在都是有效的tensor）
                inputs.labels = torch.cat(all_labels, dim=0)
                print(f"[DEBUG] 合并了{len(all_labels)}个labels，形状: {inputs.labels.shape}")
                
                # 移动到正确的设备
                inputs.input_ids = inputs.input_ids.to(images_clip.device)
                inputs.attention_mask = inputs.attention_mask.to(images_clip.device)
                inputs.pixel_values = inputs.pixel_values.to(images_clip.device)
                inputs.image_grid_thw = inputs.image_grid_thw.to(images_clip.device)
                if inputs.labels is not None:
                    inputs.labels = inputs.labels.to(images_clip.device)
                
                print(f"[DEBUG] processor生成的image_grid_thw: {inputs.image_grid_thw}")
                print(f"[DEBUG] processor生成的pixel_values形状: {inputs.pixel_values.shape}")
                print(f"[DEBUG] processor生成的input_ids形状: {inputs.input_ids.shape}")
                print(f"[DEBUG] processor生成的labels形状: {inputs.labels.shape if inputs.labels is not None else 'None'}")
                print(f"[DEBUG] 是否有图像token: {self.processor.image_token in text}")
                print(f"[DEBUG] 图像token: {self.processor.image_token}")
                print(f"[DEBUG] input_ids内容: {inputs.input_ids[0] if inputs.input_ids is not None else 'None'}")
                
                # 使用processor处理后的输入调用模型
                output = self.model.model.forward(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=inputs.image_grid_thw,  # 添加image_grid_thw参数
                    labels=inputs.labels,  # 使用processor生成的labels
                    output_hidden_states=True,
                )
                print(f"[DEBUG] 成功使用processor方法")
                
                # 更新变量以匹配processor处理后的数据
                new_input_ids = inputs.input_ids
                new_attention_mask = inputs.attention_mask
                new_labels = inputs.labels
                

            output_hidden_states = output.hidden_states
            
            # 调试：检查模型输出的维度
            print(f"[DEBUG] 模型输出分析:")
            print(f"  output类型: {type(output)}")
            print(f"  output.logits形状: {output.logits.shape if hasattr(output, 'logits') else '无logits'}")
            print(f"  output.hidden_states长度: {len(output_hidden_states) if output_hidden_states else '无hidden_states'}")
            if output_hidden_states:
                print(f"  output_hidden_states[-1]形状: {output_hidden_states[-1].shape}")
            print(f"  input_ids形状: {input_ids.shape}")
            print(f"  attention_mask形状: {attention_mask.shape}")
            print(f"  image_grid_thw: {image_grid_thw}")

        # LISA方法：通过MLP投影层处理隐藏状态 (γ: h̃_seg → h_seg)
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        
        # 调试：检查MLP投影层的输入输出维度
        print(f"[DEBUG] MLP投影层分析:")
        print(f"  text_hidden_fcs[0]输入维度: {self.model.text_hidden_fcs[0][0].in_features}")
        print(f"  text_hidden_fcs[0]输出维度: {self.model.text_hidden_fcs[0][2].out_features}")
        print(f"  output_hidden_states[-1]形状: {output_hidden_states[-1].shape}")
        
        # 这里的text_hidden_fcs[0]就是LISA论文中的MLP投影层γ
        # 确保text_hidden_fcs在正确的设备和数据类型上
        device = output_hidden_states[-1].device
        dtype = output_hidden_states[-1].dtype
        self.model.text_hidden_fcs = self.model.text_hidden_fcs.to(device=device, dtype=dtype)
        projected_hidden = self.model.text_hidden_fcs[0](output_hidden_states[-1])
        print(f"  MLP投影后形状: {projected_hidden.shape}")
        
        hidden_states.append(projected_hidden)
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        print(f"[DEBUG] LISA - MLP投影后的隐藏状态 h_seg: {last_hidden_state.shape}")

        # LISA方法：提取[SEG] token的最后一层隐藏嵌入 (h̃_seg)
        # 统一使用processor处理后的input_ids来创建seg_token_mask，避免错位风险
        
        print(f"[DEBUG] 统一使用processor输出的token序列:")
        print(f"  last_hidden_state形状: {last_hidden_state.shape}")
        print(f"  processor处理后的input_ids形状: {new_input_ids.shape}")
        print(f"  seg_token_idx: {self.seg_token_idx}")
        
        # 直接使用processor处理后的input_ids创建seg_token_mask
        # 这确保了seg_token_mask与last_hidden_state完全对齐
        seg_token_mask = new_input_ids == self.seg_token_idx
        print(f"  [DEBUG] 使用processor输出的input_ids创建seg_token_mask")
        print(f"  [DEBUG] seg_token_mask形状: {seg_token_mask.shape}")
        print(f"  [DEBUG] seg_token_mask中True的数量: {seg_token_mask.sum()}")
        
        # 验证维度对齐
        if seg_token_mask.shape != last_hidden_state.shape[:2]:
            print(f"  [ERROR] 维度不匹配: seg_token_mask {seg_token_mask.shape} vs last_hidden_state {last_hidden_state.shape[:2]}")
            print(f"  [WARNING] 跳过这个样本，不创建假掩码")
            return None, []
        else:
            print(f"  [DEBUG] 维度对齐成功")
        
        print(f"  最终seg_token_mask形状: {seg_token_mask.shape}")
        print(f"  最终seg_token_mask中True的数量: {seg_token_mask.sum()}")
        
        # 提取[SEG] token的隐藏状态 - 按batch分别处理
        pred_embeddings = []
        for i in range(seg_token_mask.shape[0]):  # 遍历batch中的每个样本
            sample_seg_mask = seg_token_mask[i]  # 获取第i个样本的[SEG] token mask
            if sample_seg_mask.any():  # 如果这个样本有[SEG] token
                sample_embeddings = last_hidden_state[i][sample_seg_mask]  # 提取该样本的[SEG] token嵌入
                pred_embeddings.append(sample_embeddings)
            else:
                # 如果这个样本没有[SEG] token，创建一个空的tensor
                print(f"[DEBUG] 样本{i}没有[SEG] token")
        
        print(f"[DEBUG] LISA - 提取[SEG] token嵌入，共{len(pred_embeddings)}个样本")
        for i, emb in enumerate(pred_embeddings):
            print(f"  样本{i}: {emb.shape}")

        # 生成mask预测 - 按照原版GLOVER的方式
        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            # 跳过空的embeddings
            if pred_embeddings[i].shape[0] == 0:
                print(f"[DEBUG] 跳过样本{i}，因为没有[SEG] token")
                continue
                
            # 确保visual_model1在正确的设备上（精度已在初始化时设置）
            device = pred_embeddings[i].device
            
            # 只在设备不匹配时才移动模型（避免频繁.to()调用）
            if not hasattr(self.model, '_sam_device_set') or self.model._sam_device_set != device:
                print(f"[DEBUG] 移动SAM模型到设备: {device}")
                self.model.visual_model1 = self.model.visual_model1.to(device=device)
                self.model._sam_device_set = device
            
            # 确保image_embeddings也在正确的设备上
            if image_embeddings[i].device != device:
                image_embeddings[i] = image_embeddings[i].to(device=device, dtype=dtype)
                print(f"[DEBUG] 移动image_embeddings[{i}]从{image_embeddings[i].device}到{device}")
            
            # 第一阶段SAM：使用visual_model1（SAM模型）
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model1.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            # 确保sparse_embeddings和dense_embeddings都在正确的设备上
            sparse_embeddings = sparse_embeddings.to(device=device, dtype=dtype)
            dense_embeddings = dense_embeddings.to(device=device, dtype=dtype)
            
            # 调试：检查所有变量的设备
            print(f"[DEBUG] 设备检查 - 样本 {i}:")
            print(f"  pred_embeddings[i]设备: {pred_embeddings[i].device}")
            print(f"  image_embeddings[i]设备: {image_embeddings[i].device}")
            print(f"  sparse_embeddings设备: {sparse_embeddings.device}")
            print(f"  dense_embeddings设备: {dense_embeddings.device}")
            print(f"  target_device: {device}")
            
            # 第一阶段SAM：使用visual_model1
            # 确保image_pe在正确的设备上
            image_pe = self.model.visual_model1.prompt_encoder.get_dense_pe()
            if image_pe.device != device:
                image_pe = image_pe.to(device=device, dtype=dtype)
            
            low_res_masks, iou_predictions = self.model.visual_model1.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            if use_text_emb_in_suffix_sam:
                text_embeds_val = pred_embeddings[i].unsqueeze(1)
            else:
                text_embeds_val = None

            # 第二阶段SAM：使用visual_model1（按照原版GLOVER的方式）
            # 确保low_res_masks在正确的设备上
            if low_res_masks.device != device:
                low_res_masks = low_res_masks.to(device=device, dtype=dtype)
            
            (
                sparse_embeddings1,
                dense_embeddings1,
            ) = self.model.visual_model1.prompt_encoder(
                points=None,
                boxes=None,
                masks=low_res_masks,  # 使用第一阶段的最佳掩码
                text_embeds=text_embeds_val,
            )
            # 确保sparse_embeddings1和dense_embeddings1都在正确的设备上
            sparse_embeddings1 = sparse_embeddings1.to(device=device, dtype=dtype)
            dense_embeddings1 = dense_embeddings1.to(device=device, dtype=dtype)
            
            # 确保第二阶段SAM的image_pe在正确的设备上
            image_pe1 = self.model.visual_model1.prompt_encoder.get_dense_pe()
            if image_pe1.device != device:
                image_pe1 = image_pe1.to(device=device, dtype=dtype)
            
            low_res_masks1, iou_predictions1 = self.model.visual_model1.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=image_pe1,
                sparse_prompt_embeddings=sparse_embeddings1,
                dense_prompt_embeddings=dense_embeddings1,
                multimask_output=multimask_output,
            )

            # postprocess_masks是一个方法，不需要移动到设备上
            
            pred_mask = self.model.visual_model1.postprocess_masks(
                low_res_masks1,
                input_size=resize_list[i],
                original_size=masks_list[i].shape[1:],
            )
        # 只取第一个掩码，因为ground truth只有一个掩码
        pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        # 计算损失
        output = model_output.logits
        
        # 使用LISA的"embedding-as-mask"范式：让模型自然学会生成[SEG] token
        if not inference and model_output.loss is not None:
            # 直接使用模型自带的损失，这通常更稳定
            ce_loss = model_output.loss
            print(f"[DEBUG] 使用模型自带CE Loss: {ce_loss.item():.6f}")
            
            # 移除[SEG] token的高权重loss，避免过拟合特定token
            # 让模型自然学习所有token，而不是强制优化[SEG] token
        else:
            # 如果没有模型损失，使用0（避免训练中断）
            ce_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
            print(f"[DEBUG] 警告: 模型没有返回损失，使用0")
        
        if self.ce_loss_weight is not None:
            ce_loss = ce_loss * self.ce_loss_weight

        # 计算mask loss
        # 获取正确的设备和数据类型
        if hasattr(output, 'logits'):
            device = output.logits.device
            dtype = output.logits.dtype
        else:
            # 如果output是tensor，使用其设备和数据类型
            device = output.device
            dtype = output.dtype
        
        mask_focal_loss = torch.tensor(0.0, device=device, dtype=dtype)
        num_masks = 0
        kl_loss = torch.tensor(0.0, device=device, dtype=dtype)
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            # 检查空的掩码（没有[SEG] token的样本）
            if pred_mask.shape[0] == 0:
                print(f"[WARNING] batch_idx {batch_idx}没有生成掩码，这可能是数据问题或模型问题。请检查：")
                print(f"  - 输入文本是否包含[SEG] token")
                print(f"  - 模型是否正确生成了分割token")
                print(f"  - 数据预处理是否正确")
                # 不直接跳过，而是记录警告并继续，让训练过程暴露这个问题
                continue

            # 检查掩码形状是否匹配，不匹配则跳过
            if gt_mask.shape != pred_mask.shape:
                print(f"[WARNING] 掩码形状不匹配，跳过: gt={gt_mask.shape}, pred={pred_mask.shape}")
                continue
                gt_mask = (gt_mask > 0.5).float()

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            
            # 使用focal loss
            mask_focal_loss += (
                sigmoid_focal_loss(pred_mask, gt_mask, num_boxes=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
            # 按照GLOVER_plus的方式计算KL loss
            kl_loss += cal_kl(pred_mask, gt_mask) * gt_mask.shape[0]
            
            # 输出KL loss的详细信息（仅第一个batch）
            if batch_idx == 0:
                print(f"[DEBUG] KL Loss 分析 (batch {batch_idx}):")
                print(f"  pred_mask shape: {pred_mask.shape}")
                print(f"  gt_mask shape: {gt_mask.shape}")
                print(f"  pred_mask range: [{pred_mask.min().item():.4f}, {pred_mask.max().item():.4f}]")
                print(f"  gt_mask range: [{gt_mask.min().item():.4f}, {gt_mask.max().item():.4f}]")
                print(f"  pred_mask sum: {pred_mask.sum().item():.4f}")
                print(f"  gt_mask sum: {gt_mask.sum().item():.4f}")
                print(f"  KL raw value: {cal_kl(pred_mask, gt_mask).item():.6f}")

        # 检查是否有太多空mask，这可能表示数据或模型问题
        total_batches = len(pred_masks)
        empty_batches = total_batches - num_masks
        if empty_batches > 0:
            empty_ratio = empty_batches / total_batches
            print(f"[WARNING] 批次中有{empty_batches}/{total_batches}个样本没有生成掩码 (比例: {empty_ratio:.2%})")
            if empty_ratio > 0.5:  # 如果超过50%的样本没有掩码，抛出异常
                raise ValueError(f"超过50%的样本没有生成掩码，这通常表示严重的数据或模型问题。请检查数据预处理和模型配置。")

        mask_loss = mask_focal_loss * 0.1 / (num_masks + 1e-8)  # 与GLOVER_plus保持一致
        # KL loss按GLOVER_plus的方式计算
        kl_loss = kl_loss / (num_masks + 1e-8)
        if self.kl_loss_weight is not None:
            kl_loss = kl_loss * self.kl_loss_weight

        # 总损失
        loss = ce_loss + mask_loss + kl_loss
        
        # 调试：输出各损失组件的详细信息
        if not inference:
            print(f"[DEBUG] 损失组件分析:")
            print(f"  CE Loss: {ce_loss.item():.6f} (权重: {self.ce_loss_weight})")
            print(f"  Mask Loss: {mask_loss.item():.6f}")
            print(f"  KL Loss: {kl_loss.item():.6f} (权重: {self.kl_loss_weight})")
            print(f"  总损失: {loss.item():.6f}")
            
            # 计算各组件在总损失中的比例
            total_abs = abs(ce_loss.item()) + abs(mask_loss.item()) + abs(kl_loss.item())
            if total_abs > 0:
                ce_ratio = abs(ce_loss.item()) / total_abs * 100
                mask_ratio = abs(mask_loss.item()) / total_abs * 100
                kl_ratio = abs(kl_loss.item()) / total_abs * 100
                print(f"  损失比例 - CE: {ce_ratio:.1f}%, Mask: {mask_ratio:.1f}%, KL: {kl_ratio:.1f}%")

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_loss": mask_loss,
            "kl_loss": kl_loss,
        }
    
    def compute_seg_token_loss(self, logits, input_ids, seg_token_idx):
        """
        LISA风格的[SEG] token损失计算
        专门用于训练模型学会生成[SEG] token
        """
        if seg_token_idx is None:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # 找到[SEG] token的位置
        seg_token_mask = input_ids == seg_token_idx
        
        if not seg_token_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # 对齐长度
        if logits.shape[1] != input_ids.shape[1]:
            min_len = min(logits.shape[1], input_ids.shape[1])
            logits = logits[:, :min_len]
            input_ids = input_ids[:, :min_len]
            seg_token_mask = seg_token_mask[:, :min_len]
        
        # 计算[SEG] token位置的损失
        seg_logits = logits[seg_token_mask]
        seg_targets = input_ids[seg_token_mask]
        
        if len(seg_logits) == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # 使用交叉熵损失
        seg_loss = F.cross_entropy(seg_logits, seg_targets, reduction='mean')
        
        return seg_loss

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs 

        # 计算mask loss
        # 获取正确的设备和数据类型
        if hasattr(output, 'logits'):
            device = output.logits.device
            dtype = output.logits.dtype
        else:
            # 如果output是tensor，使用其设备和数据类型
            device = output.device
            dtype = output.dtype
        
        mask_focal_loss = torch.tensor(0.0, device=device, dtype=dtype)
        num_masks = 0
        kl_loss = torch.tensor(0.0, device=device, dtype=dtype)
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            # 检查空的掩码（没有[SEG] token的样本）
            if pred_mask.shape[0] == 0:
                print(f"[WARNING] batch_idx {batch_idx}没有生成掩码，这可能是数据问题或模型问题。请检查：")
                print(f"  - 输入文本是否包含[SEG] token")
                print(f"  - 模型是否正确生成了分割token")
                print(f"  - 数据预处理是否正确")
                # 不直接跳过，而是记录警告并继续，让训练过程暴露这个问题
                continue

            # 检查掩码形状是否匹配，不匹配则跳过
            if gt_mask.shape != pred_mask.shape:
                print(f"[WARNING] 掩码形状不匹配，跳过: gt={gt_mask.shape}, pred={pred_mask.shape}")
                continue
                gt_mask = (gt_mask > 0.5).float()

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            
            # 使用focal loss
            mask_focal_loss += (
                sigmoid_focal_loss(pred_mask, gt_mask, num_boxes=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]
            # 按照GLOVER_plus的方式计算KL loss
            kl_loss += cal_kl(pred_mask, gt_mask) * gt_mask.shape[0]
            
            # 输出KL loss的详细信息（仅第一个batch）
            if batch_idx == 0:
                print(f"[DEBUG] KL Loss 分析 (batch {batch_idx}):")
                print(f"  pred_mask shape: {pred_mask.shape}")
                print(f"  gt_mask shape: {gt_mask.shape}")
                print(f"  pred_mask range: [{pred_mask.min().item():.4f}, {pred_mask.max().item():.4f}]")
                print(f"  gt_mask range: [{gt_mask.min().item():.4f}, {gt_mask.max().item():.4f}]")
                print(f"  pred_mask sum: {pred_mask.sum().item():.4f}")
                print(f"  gt_mask sum: {gt_mask.sum().item():.4f}")
                print(f"  KL raw value: {cal_kl(pred_mask, gt_mask).item():.6f}")

        # 检查是否有太多空mask，这可能表示数据或模型问题
        total_batches = len(pred_masks)
        empty_batches = total_batches - num_masks
        if empty_batches > 0:
            empty_ratio = empty_batches / total_batches
            print(f"[WARNING] 批次中有{empty_batches}/{total_batches}个样本没有生成掩码 (比例: {empty_ratio:.2%})")
            if empty_ratio > 0.5:  # 如果超过50%的样本没有掩码，抛出异常
                raise ValueError(f"超过50%的样本没有生成掩码，这通常表示严重的数据或模型问题。请检查数据预处理和模型配置。")

        if num_masks > 0:
            mask_loss = mask_focal_loss * 0.1 / num_masks  # 与GLOVER_plus保持一致
        else:
            mask_loss = mask_focal_loss  # 已经是tensor(0.0)
        # KL loss按GLOVER_plus的方式计算
        kl_loss = kl_loss / (num_masks + 1e-8)
        if self.kl_loss_weight is not None:
            kl_loss = kl_loss * self.kl_loss_weight

        # 总损失
        loss = ce_loss + mask_loss + kl_loss
        
        # 调试：输出各损失组件的详细信息
        if not inference:
            print(f"[DEBUG] 损失组件分析:")
            print(f"  CE Loss: {ce_loss.item():.6f} (权重: {self.ce_loss_weight})")
            print(f"  Mask Loss: {mask_loss.item():.6f}")
            print(f"  KL Loss: {kl_loss.item():.6f} (权重: {self.kl_loss_weight})")
            print(f"  总损失: {loss.item():.6f}")
            
            # 计算各组件在总损失中的比例
            total_abs = abs(ce_loss.item()) + abs(mask_loss.item()) + abs(kl_loss.item())
            if total_abs > 0:
                ce_ratio = abs(ce_loss.item()) / total_abs * 100
                mask_ratio = abs(mask_loss.item()) / total_abs * 100
                kl_ratio = abs(kl_loss.item()) / total_abs * 100
                print(f"  损失比例 - CE: {ce_ratio:.1f}%, Mask: {mask_ratio:.1f}%, KL: {kl_ratio:.1f}%")

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_loss": mask_loss,
            "kl_loss": kl_loss,
        }
    
    def compute_seg_token_loss(self, logits, input_ids, seg_token_idx):
        """
        LISA风格的[SEG] token损失计算
        专门用于训练模型学会生成[SEG] token
        """
        if seg_token_idx is None:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # 找到[SEG] token的位置
        seg_token_mask = input_ids == seg_token_idx
        
        if not seg_token_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # 对齐长度
        if logits.shape[1] != input_ids.shape[1]:
            min_len = min(logits.shape[1], input_ids.shape[1])
            logits = logits[:, :min_len]
            input_ids = input_ids[:, :min_len]
            seg_token_mask = seg_token_mask[:, :min_len]
        
        # 计算[SEG] token位置的损失
        seg_logits = logits[seg_token_mask]
        seg_targets = input_ids[seg_token_mask]
        
        if len(seg_logits) == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # 使用交叉熵损失
        seg_loss = F.cross_entropy(seg_logits, seg_targets, reduction='mean')
        
        return seg_loss

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs 