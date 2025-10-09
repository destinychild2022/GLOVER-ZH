from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel
import math
from typing import Tuple, Type

from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)

from .llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
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


# ==================== LISA Attention Components ====================

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class LISA_TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

    def forward(
        self, queries: Tensor, keys: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        attn_out = self.self_attn(q=queries, k=queries, v=queries)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        attn_out = self.cross_attn_token_to_image(q=queries, k=keys, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        attn_out = self.cross_attn_image_to_token(q=keys, k=queries, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


# ==================== GLOVER_LISA Model ====================

class GloverLISAMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GloverLISAMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_glover_lisa_modules(self.config)

    def initialize_glover_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # DINO-V2 (新增)
        try:
            dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.visual_model_dinov2 = dinov2_vitl14
            for param in self.visual_model_dinov2.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load DINOv2: {e}")
            self.visual_model_dinov2 = None

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

        # LISA-specific modules (新增)
        if self.visual_model_dinov2 is not None:
            # 1x1 conv to reduce the dimension of the image feature to 256
            self.lisa_dino_conv = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)

            self.lisa_attention_layers = nn.ModuleList()
            depth = 2
            for i in range(depth):
                self.lisa_attention_layers.append(
                    LISA_TwoWayAttentionBlock(
                        embedding_dim=256,
                        num_heads=8,
                        mlp_dim=2048,
                        attention_downsample_rate=1,
                    )
                )
            self.lisa_final_attn = Attention(
                embedding_dim=256, num_heads=8, downsample_rate=1
            )
            self.lisa_norm_final_attn = nn.LayerNorm(256)

            self.lisa_iou_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

            self.lisa_embedding_head = nn.Sequential(
                nn.Linear(256, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 256),
            )


class GloverLISAModel(GloverLISAMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GloverLISAModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class GloverLISAForCausalLM(LlavaLlamaForCausalLM):
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
            # 新增LISA损失权重
            self.align_loss_weight = kwargs.pop("align_loss_weight", 1.0)
            self.regression_loss_weight = kwargs.pop("regression_loss_weight", 1.0)
        else:
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.kl_loss_weight = kwargs.pop("kl_loss_weight", None)
            # 新增LISA损失权重
            self.align_loss_weight = kwargs.pop("align_loss_weight", 1.0)
            self.regression_loss_weight = kwargs.pop("regression_loss_weight", 1.0)
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx", 1000)

        super().__init__(config)

        self.model = GloverLISAModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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

    def get_dinov2_visual_embs(self, pixel_values: torch.FloatTensor):
        """获取DINOv2视觉特征"""
        if self.model.visual_model_dinov2 is None:
            return None
            
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings_dict = self.model.visual_model_dinov2.forward_features(pixel_values[i].unsqueeze(0))
                image_embeddings = image_embeddings_dict['x_norm_patchtokens']
                # 1*4096*1024 -> 1*1024*64*64
                image_embeddings = image_embeddings.permute(0, 2, 1).reshape(1, 1024, 64, 64)
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def mask_pooling(self, image_embeddings: torch.FloatTensor, weight_maps: torch.FloatTensor):
        """
        LISA的掩码池化操作
        Args:
            image_embeddings: [256, 64, 64] - DINOv2特征
            weight_maps: [K, 64, 64] - SAM掩码
        Returns:
            output: [K, 256] - 每个掩码的特征向量
        """
        # [256, 64, 64] -> [256, 4096]
        image_embeddings = image_embeddings.flatten(1, 2)
        # [K, 64, 64] -> [K, 4096]
        weight_maps = weight_maps.flatten(1, 2)
        # [K, 4096] -> [K, 256]
        output = weight_maps @ image_embeddings.T
        # normalize
        output = output / (weight_maps.sum(-1, keepdim=True) + 1e-8)

        assert output.shape[0] == weight_maps.shape[0]
        assert output.shape[1] == image_embeddings.shape[0]

        return output

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
        sam_segs_list: List[torch.FloatTensor] = None,  # 新增SAM掩码列表
        sam_ious_list: List[torch.FloatTensor] = None,  # 新增IoU列表
        sam_iops_list: List[torch.FloatTensor] = None,  # 新增IoP列表
        inference: bool = False,
        use_text_emb_in_suffix_sam: bool = False,
        **kwargs,
    ):
        # 获取图像嵌入
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # 处理segmentation token
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

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().to(input_ids.device), seg_token_offset], dim=0
        )
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        # ==================== LISA处理流程 ====================
        if sam_segs_list is not None and self.model.visual_model_dinov2 is not None:
            # 获取DINOv2特征
            dinov2_embeddings = self.get_dinov2_visual_embs(images)
            dinov2_embeddings = self.model.lisa_dino_conv(dinov2_embeddings)
            
            # 上采样到256x256
            origin_dtype = dinov2_embeddings.dtype
            dinov2_embeddings = dinov2_embeddings.to(dtype=torch.float32)
            dinov2_embeddings = F.interpolate(dinov2_embeddings, size=(256, 256), mode='bilinear', align_corners=False)
            dinov2_embeddings = dinov2_embeddings.to(dtype=origin_dtype)

            # LISA掩码池化和注意力处理
            sam_segs_feature_list = []
            sam_pred_ious_list = []
            
            for batch_idx in range(len(sam_segs_list)):
                segs = sam_segs_list[batch_idx]  # (K, 256, 256)
                segs_feature = self.mask_pooling(dinov2_embeddings[batch_idx], segs)  # (K, 256)
                
                # 获取文本特征
                text_feature = pred_embeddings[batch_idx]  # (C, 256)
                text_feature = text_feature.unsqueeze(1)  # (C, 1, 256)
                
                number_conversations = text_feature.shape[0]
                segs_feature = segs_feature.unsqueeze(0)  # (1, K, 256)
                
                if number_conversations > 0:
                    segs_feature = segs_feature.expand(number_conversations, -1, -1)  # (C, K, 256)

                # 双向注意力交互
                for layer in self.model.lisa_attention_layers:
                    segs_feature, text_feature = layer(
                        queries=segs_feature,
                        keys=text_feature,
                    )

                # 最终注意力
                attn_out = self.model.lisa_final_attn(q=segs_feature, k=text_feature, v=text_feature)
                segs_feature = segs_feature + attn_out
                segs_feature = self.model.lisa_norm_final_attn(segs_feature)

                # IoU预测和特征嵌入
                sam_iou = self.model.lisa_iou_head(segs_feature)  # (C, K, 1)
                sam_pred_ious_list.append(sam_iou)
                
                segs_feature = self.model.lisa_embedding_head(segs_feature)  # (C, K, 256)
                sam_segs_feature_list.append(segs_feature)

            # 推理时返回LISA结果
            if inference:
                pred_similarity = []
                for batch_idx in range(len(pred_embeddings)):
                    pred_embedding = pred_embeddings[batch_idx]  # (1, 256)
                    pred_embedding_normlized = pred_embedding / pred_embedding.norm(dim=-1, keepdim=True)
                    sam_features = sam_segs_feature_list[batch_idx][0, :, :]  # (K, 256)
                    sam_features_normlized = sam_features / sam_features.norm(dim=-1, keepdim=True)
                    similarity = pred_embedding_normlized @ sam_features_normlized.T  # (1, K)
                    pred_similarity.append(similarity)

                pred_ious = []
                for batch_idx in range(len(sam_pred_ious_list)):
                    sam_pred_ious = sam_pred_ious_list[batch_idx][0, :, :]  # (K, 1)
                    pred_ious.append(sam_pred_ious.T)  # (1, K)

                return {
                    "pred_similarity": pred_similarity,
                    "gt_masks": masks_list,
                    "pred_iou": pred_ious,
                }

            # 训练时的损失计算
            if sam_ious_list is not None and sam_iops_list is not None:
                # 计算LISA损失
                align_loss = 0.0
                regression_loss = 0.0
                valid_batch = 0
                
                for batch_idx in range(len(sam_segs_feature_list)):
                    segs_feature = sam_segs_feature_list[batch_idx]  # (C, K, 256)
                    gt_iou = sam_ious_list[batch_idx]  # (C, K)
                    gt_iop = sam_iops_list[batch_idx]  # (C, K)
                    pred_iou = sam_pred_ious_list[batch_idx]  # (C, K, 1)
                    
                    number_rounds = pred_embeddings[batch_idx].shape[0]
                    
                    if number_rounds == 0:
                        continue
                        
                    for round_idx in range(number_rounds):
                        gt_iou_round = gt_iou[round_idx].unsqueeze(1)  # (K, 1)
                        gt_iou_round = gt_iou_round.to(dtype=pred_iou.dtype)
                        gt_iop_round = gt_iop[round_idx].unsqueeze(1).to(dtype=pred_iou.dtype)  # (K, 1)
                        
                        # 对齐损失和回归损失
                        target_embedding = pred_embeddings[batch_idx][round_idx].unsqueeze(0)  # (1, 256)
                        # 这里需要实现softmax_align_loss和iou_regression_loss函数
                        # align_loss += softmax_align_loss(segs_feature[round_idx], target_embedding, gt_iou_round)
                        # regression_loss += iou_regression_loss(pred_iou[round_idx], gt_iop_round)
                    
                    if number_rounds > 0:
                        valid_batch += 1
                
                if valid_batch > 0:
                    align_loss = align_loss / valid_batch
                    regression_loss = regression_loss / valid_batch

        # ==================== 原始GLOVER处理流程 ====================
        else:
            # 如果没有SAM掩码或DINOv2，回退到原始GLOVER处理
            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=masks_list[i].shape[1:],
                )
                pred_masks.append(pred_mask[:, 0])

            if inference:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": masks_list,
                }

        # 计算损失
        model_output = output
        gt_masks = masks_list

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight

        mask_focal_loss = 0
        num_masks = 0
        kl_loss = 0
        
        if sam_segs_list is None:  # 原始GLOVER损失
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx]
                pred_mask = pred_masks[batch_idx]

                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                mask_focal_loss += (
                    sigmoid_focal_loss(pred_mask, gt_mask, num_boxes=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]
                kl_loss += cal_kl(pred_mask, gt_mask) * gt_mask.shape[0]

            mask_loss = mask_focal_loss * 0.1 / (num_masks + 1e-8)
            kl_loss = kl_loss / (num_masks + 1e-8)
            kl_loss = kl_loss * self.kl_loss_weight

            loss = ce_loss + mask_loss + kl_loss

            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "mask_loss": mask_loss,
                "kl_loss": kl_loss,
            }
        else:
            # LISA损失
            loss = ce_loss + self.align_loss_weight * align_loss + self.regression_loss_weight * regression_loss
            
            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "align_loss": align_loss,
                "regression_loss": regression_loss,
            }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        sam_segs_list=None,  # 新增SAM掩码参数
        max_new_tokens=32,
        tokenizer=None,
        use_text_emb_in_suffix_sam: bool = False,
    ):
        with torch.no_grad():
            try:
                outputs = super().generate(
                    input_ids=input_ids,
                    images=images_clip,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    use_cache=True,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer else None,
                    eos_token_id=tokenizer.eos_token_id if tokenizer else None,
                )
                
                if not hasattr(outputs, 'hidden_states'):
                    if isinstance(outputs, str):
                        return None, []
                    else:
                        return None, []
                
                output_hidden_states = outputs.hidden_states[-1]
                output_ids = outputs.sequences
                
            except Exception as e:
                return None, []

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().to(input_ids.device),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().to(input_ids.device), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            # 如果有SAM掩码，使用LISA处理
            if sam_segs_list is not None and self.model.visual_model_dinov2 is not None:
                # 使用LISA推理流程
                result = self.model_forward(
                    images=images,
                    images_clip=images_clip,
                    input_ids=input_ids,
                    labels=None,
                    attention_masks=None,
                    offset=torch.tensor([0, len(pred_embeddings)]).to(input_ids.device),
                    masks_list=[],
                    resize_list=resize_list,
                    sam_segs_list=sam_segs_list,
                    inference=True,
                )
                return output_ids, result.get("pred_similarity", [])
            else:
                # 使用原始GLOVER处理
                image_embeddings = self.get_visual_embs(images)
                multimask_output = False
                pred_masks = []
                
                for i in range(len(pred_embeddings)):
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i].unsqueeze(1),
                    )

                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )

                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=original_size_list[i],
                    )
                    pred_masks.append(pred_mask[:, 0])

                return output_ids, pred_masks
