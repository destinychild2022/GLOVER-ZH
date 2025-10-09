from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)

from .llava.model.language_model.llava_qwen import (
    LlavaQwenForCausalLM,
    LlavaQwenModel,
)
from .segment_anything import build_sam_vit_h
import numpy as np
import pdb


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
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
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
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
    inputs = inputs.sigmoid()
    inputs = inputs / (inputs.sum(dim=(1, 2), keepdim=True) + eps)
    targets = targets / (targets.sum(dim=(1, 2), keepdim=True) + eps)

    kld = targets * (torch.log(targets + eps) - torch.log(inputs + eps))
    kld = kld.sum()

    return kld


class GloverMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GloverMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs.get("train_mask_decoder", True)
            self.config.out_dim = kwargs.get("out_dim", 256)
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        
        # 总是初始化GLOVER模块
        self.initialize_glover_modules(self.config)

    def initialize_glover_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        self.visual_model1 = build_sam_vit_h(self.vision_pretrained)
        
        # 确保SAM模型在正确的设备上 - 使用to_empty()处理meta tensor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visual_model = self.visual_model.to_empty(device=device)
        self.visual_model1 = self.visual_model1.to_empty(device=device)
        
        # 确保所有参数都在正确的设备上
        for param in self.visual_model.parameters():
            param.requires_grad = False
        for param in self.visual_model1.parameters():
            param.requires_grad = False
            
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

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


class GloverQwenVLModel(GloverMetaModel, LlavaQwenModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GloverQwenVLModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
    
    def embed_images(self, images, inputs_embeds):
        """嵌入图像到输入嵌入中"""
        # 对于GLOVER模型，我们不需要特殊的图像嵌入处理
        # 因为图像处理是通过get_visual_embs方法完成的
        return inputs_embeds
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class GloverQwenVLForCausalLM(LlavaQwenForCausalLM):
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
            config.train_mask_decoder = kwargs.pop("train_mask_decoder", True)
            config.out_dim = kwargs.pop("out_dim", 256)
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
        
        # 设置模型路径用于Qwen2.5-VL加载
        config.model_path = config.mm_vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx", 1000)

        super().__init__(config)

        self.model = GloverQwenVLModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.model.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.model.model.set_input_embeddings(value)
    
    def gradient_checkpointing_enable(self):
        """启用梯度检查点"""
        # 对于Qwen-VL模型，我们需要在底层的语言模型上启用梯度检查点
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing_enable'):
            self.model.model.gradient_checkpointing_enable()
        else:
            # 如果底层模型不支持，我们跳过这个功能
            print("警告：底层模型不支持梯度检查点，跳过启用")
    

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            # 优化：批量处理而不是逐个处理
            if pixel_values.shape[0] > 1:
                # 批量处理
                image_embeddings = self.model.visual_model.image_encoder(pixel_values)
            else:
                # 单个样本处理
                image_embeddings_list = []
                for i in range(pixel_values.shape[0]):
                    image_embeddings = self.model.visual_model.image_encoder(
                        pixel_values[i].unsqueeze(0)
                    )
                    image_embeddings_list.append(image_embeddings)
                image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        resize_list: List[tuple],
        inference: bool = False,
        use_text_emb_in_suffix_sam: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

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
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []
        assert len(output_hidden_states) == self.config.num_hidden_layers + 1
        for i in range(self.config.num_hidden_layers + 1):
            hidden_states.append(output_hidden_states[i])

        new_hidden_states = []
        assert len(hidden_states) == self.config.num_hidden_layers + 1
        for i in range(self.config.num_hidden_layers + 1):
            if i < self.config.num_hidden_layers:
                new_hidden_states.append(hidden_states[i])
            else:
                new_hidden_states.append(hidden_states[i])

        hidden_states = new_hidden_states
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = hidden_states.permute(1, 0, 2, 3).contiguous()

        seg_token_features = hidden_states[seg_token_mask]
        seg_token_features = seg_token_features.view(
            batch_size, -1, seg_token_features.shape[-1]
        )

        if use_text_emb_in_suffix_sam:
            text_emb = seg_token_features
        else:
            text_emb = None

        seg_token_features = self.model.text_hidden_fcs[0](seg_token_features)

        seg_token_features = seg_token_features / (
            seg_token_features.norm(dim=-1, keepdim=True) + 1e-6
        )

        image_embeddings = image_embeddings / (
            image_embeddings.norm(dim=-1, keepdim=True) + 1e-6
        )

        sparse_embeddings, dense_embeddings = self.model.visual_model1.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_emb=text_emb,
        )

        low_res_masks, iou_predictions = self.model.visual_model1.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.visual_model1.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

        if inference:
            return masks

        ce_loss = 0.0
        mask_loss = 0.0
        kl_loss = 0.0

        for i in range(batch_size):
            start_i, end_i = offset[i], offset[i + 1]
            cur_masks = masks[i]
            cur_masks_list = masks_list[start_i:end_i]
            cur_resize_list = resize_list[start_i:end_i]

            for j, (cur_mask, cur_resize) in enumerate(
                zip(cur_masks_list, cur_resize_list)
            ):
                cur_mask = F.interpolate(
                    cur_mask.unsqueeze(0).unsqueeze(0).float(),
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

                if self.ce_loss_weight > 0:
                    ce_loss += sigmoid_ce_loss(
                        cur_masks[j], cur_mask, num_masks=1
                    ) * self.ce_loss_weight

                if self.dice_loss_weight > 0:
                    mask_loss += dice_loss(
                        cur_masks[j], cur_mask, num_masks=1
                    ) * self.dice_loss_weight

                if self.bce_loss_weight > 0:
                    mask_loss += sigmoid_ce_loss(
                        cur_masks[j], cur_mask, num_masks=1
                    ) * self.bce_loss_weight

                if self.kl_loss_weight > 0:
                    kl_loss += cal_kl(cur_masks[j], cur_mask) * self.kl_loss_weight

        total_loss = ce_loss + mask_loss + kl_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_loss": mask_loss,
            "kl_loss": kl_loss,
        }

    def evaluate(
        self,
        images_clip,
        input_ids,
        attention_masks,
        offset,
        masks_list,
        resize_list,
        intersectionAndUnionGPU,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images_clip)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

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

        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            output_hidden_states=True,
        )

        hidden_states = []
        assert len(output.hidden_states) == self.config.num_hidden_layers + 1
        for i in range(self.config.num_hidden_layers + 1):
            hidden_states.append(output.hidden_states[i])

        new_hidden_states = []
        assert len(hidden_states) == self.config.num_hidden_layers + 1
        for i in range(self.config.num_hidden_layers + 1):
            if i < self.config.num_hidden_layers:
                new_hidden_states.append(hidden_states[i])
            else:
                new_hidden_states.append(hidden_states[i])

        hidden_states = new_hidden_states
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = hidden_states.permute(1, 0, 2, 3).contiguous()

        seg_token_features = hidden_states[seg_token_mask]
        seg_token_features = seg_token_features.view(
            batch_size, -1, seg_token_features.shape[-1]
        )

        seg_token_features = self.model.text_hidden_fcs[0](seg_token_features)

        seg_token_features = seg_token_features / (
            seg_token_features.norm(dim=-1, keepdim=True) + 1e-6
        )

        image_embeddings = image_embeddings / (
            image_embeddings.norm(dim=-1, keepdim=True) + 1e-6
        )

        sparse_embeddings, dense_embeddings = self.model.visual_model1.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        low_res_masks, iou_predictions = self.model.visual_model1.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.visual_model1.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

        intersection, union, target = intersectionAndUnionGPU(
            masks, masks_list, resize_list, offset
        )

        return intersection, union, target
