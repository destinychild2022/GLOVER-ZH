import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_align_loss(segs_feature, target_embedding, gt_iou):
    """
    LISA的对齐损失函数
    Args:
        segs_feature: (K, D) - 掩码特征
        target_embedding: (1, D) - 目标文本嵌入
        gt_iou: (K, 1) - 真实IoU分数
    Returns:
        loss: 对齐损失
    """
    # 计算相似度
    similarity = segs_feature @ target_embedding.T  # (K, 1)
    
    # 使用softmax计算注意力权重
    attention_weights = F.softmax(similarity, dim=0)  # (K, 1)
    
    # 计算加权IoU
    weighted_iou = (attention_weights * gt_iou).sum()
    
    # 对齐损失：希望注意力权重与IoU分数对齐
    align_loss = -weighted_iou
    
    return align_loss


def iou_regression_loss(pred_iou, gt_iop):
    """
    IoU回归损失函数
    Args:
        pred_iou: (K, 1) - 预测的IoU分数
        gt_iop: (K, 1) - 真实的IoP分数
    Returns:
        loss: 回归损失
    """
    # 使用MSE损失
    regression_loss = F.mse_loss(pred_iou, gt_iop)
    
    return regression_loss


def sigmoid_align_loss(segs_feature, target_embedding, gt_iou, temperature=10.0, bias=-10.0):
    """
    使用sigmoid的对齐损失函数（备选实现）
    Args:
        segs_feature: (K, D) - 掩码特征
        target_embedding: (1, D) - 目标文本嵌入
        gt_iou: (K, 1) - 真实IoU分数
        temperature: 温度参数
        bias: 偏置参数
    Returns:
        loss: 对齐损失
    """
    # 计算相似度
    similarity = segs_feature @ target_embedding.T  # (K, 1)
    
    # 应用温度和偏置
    logits = temperature * similarity + bias
    
    # 使用sigmoid激活
    probs = torch.sigmoid(logits)  # (K, 1)
    
    # 计算加权IoU
    weighted_iou = (probs * gt_iou).sum()
    
    # 对齐损失
    align_loss = -weighted_iou
    
    return align_loss


def contrastive_align_loss(segs_feature, target_embedding, gt_iou, margin=0.5):
    """
    对比学习对齐损失函数（备选实现）
    Args:
        segs_feature: (K, D) - 掩码特征
        target_embedding: (1, D) - 目标文本嵌入
        gt_iou: (K, 1) - 真实IoU分数
        margin: 边界参数
    Returns:
        loss: 对比损失
    """
    # 计算相似度
    similarity = segs_feature @ target_embedding.T  # (K, 1)
    
    # 创建正负样本标签
    positive_mask = gt_iou > 0.5  # 高IoU为正样本
    negative_mask = gt_iou <= 0.5  # 低IoU为负样本
    
    # 正样本损失
    positive_loss = (1 - similarity) * positive_mask
    positive_loss = positive_loss.sum() / (positive_mask.sum() + 1e-8)
    
    # 负样本损失
    negative_loss = torch.clamp(similarity - margin, min=0) * negative_mask
    negative_loss = negative_loss.sum() / (negative_mask.sum() + 1e-8)
    
    # 总损失
    contrastive_loss = positive_loss + negative_loss
    
    return contrastive_loss


def focal_align_loss(segs_feature, target_embedding, gt_iou, alpha=0.25, gamma=2.0):
    """
    使用Focal Loss的对齐损失函数（备选实现）
    Args:
        segs_feature: (K, D) - 掩码特征
        target_embedding: (1, D) - 目标文本嵌入
        gt_iou: (K, 1) - 真实IoU分数
        alpha: 平衡参数
        gamma: 聚焦参数
    Returns:
        loss: Focal对齐损失
    """
    # 计算相似度
    similarity = segs_feature @ target_embedding.T  # (K, 1)
    
    # 将IoU转换为二分类标签
    labels = (gt_iou > 0.5).float()  # (K, 1)
    
    # 计算概率
    probs = torch.sigmoid(similarity)  # (K, 1)
    
    # Focal Loss
    ce_loss = F.binary_cross_entropy_with_logits(similarity, labels, reduction='none')
    p_t = probs * labels + (1 - probs) * (1 - labels)
    focal_weight = alpha * (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss
    
    return focal_loss.mean()


class LISALoss(nn.Module):
    """
    LISA损失函数的封装类
    """
    def __init__(self, align_loss_type='softmax', regression_loss_type='mse', 
                 align_loss_weight=1.0, regression_loss_weight=1.0):
        super().__init__()
        self.align_loss_type = align_loss_type
        self.regression_loss_type = regression_loss_type
        self.align_loss_weight = align_loss_weight
        self.regression_loss_weight = regression_loss_weight
        
    def forward(self, segs_feature, target_embedding, pred_iou, gt_iou, gt_iop):
        """
        计算LISA损失
        Args:
            segs_feature: (K, D) - 掩码特征
            target_embedding: (1, D) - 目标文本嵌入
            pred_iou: (K, 1) - 预测IoU
            gt_iou: (K, 1) - 真实IoU
            gt_iop: (K, 1) - 真实IoP
        Returns:
            total_loss: 总损失
            align_loss: 对齐损失
            regression_loss: 回归损失
        """
        # 对齐损失
        if self.align_loss_type == 'softmax':
            align_loss = softmax_align_loss(segs_feature, target_embedding, gt_iou)
        elif self.align_loss_type == 'sigmoid':
            align_loss = sigmoid_align_loss(segs_feature, target_embedding, gt_iou)
        elif self.align_loss_type == 'contrastive':
            align_loss = contrastive_align_loss(segs_feature, target_embedding, gt_iou)
        elif self.align_loss_type == 'focal':
            align_loss = focal_align_loss(segs_feature, target_embedding, gt_iou)
        else:
            raise ValueError(f"Unknown align loss type: {self.align_loss_type}")
        
        # 回归损失
        if self.regression_loss_type == 'mse':
            regression_loss = iou_regression_loss(pred_iou, gt_iop)
        else:
            raise ValueError(f"Unknown regression loss type: {self.regression_loss_type}")
        
        # 总损失
        total_loss = (self.align_loss_weight * align_loss + 
                     self.regression_loss_weight * regression_loss)
        
        return total_loss, align_loss, regression_loss
