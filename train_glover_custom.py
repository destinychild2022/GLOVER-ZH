# 设置环境变量解决bitsandbytes权限问题
import os
import tempfile

# 创建临时目录作为HOME
temp_home = tempfile.mkdtemp()
os.environ['HOME'] = temp_home
os.environ['USERPROFILE'] = temp_home  # Windows兼容
os.environ['XDG_CONFIG_HOME'] = os.path.join(temp_home, '.config')
os.environ['XDG_CACHE_HOME'] = os.path.join(temp_home, '.cache')

# 设置bitsandbytes相关环境变量
os.environ['BITSANDBYTES_FUNCTIONAL'] = '1'
os.environ['BITSANDBYTES_CUDA_SETUP'] = '0'
os.environ['BITSANDBYTES_NO_CUDA'] = '1'  # 禁用CUDA相关检查
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  # 确保CUDA设备可见

# 设置其他可能需要的环境变量
os.environ['NVM_DIR'] = os.path.join(temp_home, '.nvm')
os.environ['NODE_PATH'] = os.path.join(temp_home, '.node')

# 创建必要的目录
os.makedirs(os.environ['XDG_CONFIG_HOME'], exist_ok=True)
os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)
os.makedirs(os.environ['NVM_DIR'], exist_ok=True)

# 设置SwanLab API Key环境变量
os.environ['SWANLAB_API_KEY'] = 'BBd5HKuM6sIhTwyWmgZ6Z'

# 设置CUDA内存优化环境变量（移除expandable_segments）
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import argparse
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import logging
import swanlab

from model.GLOVER_plus import GloverForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, collate_fn
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    AverageMeter,
    ProgressMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_glover_custom.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def parse_args(args):
    parser = argparse.ArgumentParser(description="GLOVER++ Custom Annotation Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size for SAM")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower",
        default="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2",
        type=str,
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--use_text_emb_in_suffix_sam", action="store_true", default=False
    )
    parser.add_argument("--train_firs_mask_decoder", action="store_true", default=False)
    parser.add_argument(
        "--use_diff_lr",
        action="store_true",
        default=False,
        help="whether to use different lr for mask decoder and aff decoder",
    )
    parser.add_argument(
        "--train_sec_prompt_encoder", action="store_true", default=False
    )
    parser.add_argument(
        "--lr_ratio",
        default=0.1,
        type=float,
        help="lr ratio between mask decoder and aff decoder",
    )
    parser.add_argument("--dataset", default="custom_annotation", type=str)
    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument(
        "--dataset_dir", default="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation/non_grasp_frames_json/images",
        type=str
    )
    parser.add_argument(
        "--annotation_dir", default="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation/output",
        type=str
    )
    parser.add_argument(
        "--sam_vit_path",
        default="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth",
        type=str,
    )
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="glover++_custom_annotation", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=128, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=8,
        type=int,
    )
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--ce_loss_weight", default=0.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--kl_loss_weight", default=0.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default=None, type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU memory cleared before training")
    
    # 初始化Swanlab
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        
        # 启动Swanlab实验
        swanlab.init(
            experiment_name=args.exp_name,
            config={
                "model": "GLOVER++",
                "dataset": "custom_annotation",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "precision": args.precision,
            },
            api_key=os.environ['SWANLAB_API_KEY']
        )
        logger.info("Swanlab experiment initialized")
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token  # <unk>
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "kl_loss_weight": args.kl_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = GloverForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    suffix_sam_weight = torch.load(args.sam_vit_path, map_location="cpu")
    model.get_model().visual_model1.load_state_dict(suffix_sam_weight, strict=True)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    # LoRA配置
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "visual_model1",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    model.resize_token_embeddings(len(tokenizer))
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in [
                    "lora_A",
                    "lora_B",
                    "lm_head",
                    "embed_tokens",
                    "text_hidden_fcs",
                ]
            ]
        ):
            p.requires_grad = not (args.ce_loss_weight == 0.0)

    for n, p in model.named_parameters():
        if "visual_model1." in n:
            if "prompt_encoder" in n:
                p.requires_grad = args.train_sec_prompt_encoder
            elif "image_encoder" in n:
                p.requires_grad = False
            elif "mask_decoder" in n:
                p.requires_grad = True
        if "visual_model." in n:
            p.requires_grad = False

    if args.train_firs_mask_decoder:
        for n, p in model.named_parameters():
            if "visual_model." and "mask_decoder" in n:
                p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    # 创建自定义数据集
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        annotation_dir=args.annotation_dir,  # 添加annotation_dir参数
    )
    logger.info(f"Training with {len(train_dataset)} examples from custom annotation dataset.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.steps_per_epoch * args.epochs // 10,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "allgather_bucket_size": 5e7,
            "cpu_offload": False,
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
    }

    if args.use_diff_lr:
        param_groups = []
        visual_model_params = []
        visual_model1_params = []
        other_params = []

        for n, p in model.named_parameters():
            if "visual_model." in n and p.requires_grad:
                visual_model_params.append(p)
            elif "visual_model1." in n and p.requires_grad:
                visual_model1_params.append(p)
            elif p.requires_grad:
                other_params.append(p)

        if visual_model_params:
            param_groups.append(
                {"params": visual_model_params, "lr": args.lr * args.lr_ratio}
            )
        if visual_model1_params:
            param_groups.append({"params": visual_model1_params, "lr": args.lr})
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr})

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters() if not args.use_diff_lr else param_groups,
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        logger.info(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0
    
    # 记录训练开始时间
    training_start_time = time.time()
    total_epochs = args.epochs - args.start_epoch
    
    if args.local_rank == 0:
        logger.info(f"Starting training for {total_epochs} epochs from epoch {args.start_epoch}")
        logger.info(f"Total steps: {args.epochs * args.steps_per_epoch}")
        logger.info(f"Steps per epoch: {args.steps_per_epoch}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Total batch size: {args.batch_size * args.grad_accumulation_steps * torch.cuda.device_count()}")

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        if args.local_rank == 0:
            logger.info(f"Starting Epoch [{epoch}/{args.epochs-1}]")
        
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        # 计算epoch完成时间和预测剩余时间
        epoch_time = time.time() - epoch_start_time
        remaining_epochs = args.epochs - epoch - 1
        estimated_remaining_time = remaining_epochs * epoch_time
        
        if args.local_rank == 0:
            logger.info(f"Epoch [{epoch}] completed in {format_time(epoch_time)}")
            if remaining_epochs > 0:
                logger.info(f"Estimated remaining time: {format_time(estimated_remaining_time)} ({remaining_epochs} epochs left)")

        save_dir = os.path.join(args.log_dir, "ckpt_model")
        if args.local_rank == 0:
            torch.save(
                {"epoch": epoch},
                os.path.join(
                    args.log_dir,
                    "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(best_score, cur_ciou),
                ),
            )
            if os.path.exists(save_dir):
                import shutil
                shutil.rmtree(save_dir)
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)
    
    # 记录总训练时间
    if args.local_rank == 0:
        total_training_time = time.time() - training_start_time
        logger.info(f"Training completed! Total training time: {format_time(total_training_time)}")
        logger.info(f"Average time per epoch: {format_time(total_training_time / total_epochs)}")


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    kl_losses = AverageMeter("KlLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            kl_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    epoch_start_time = time.time()
    
    # 计算总训练步数
    total_steps = args.epochs * args.steps_per_epoch
    current_global_step = epoch * args.steps_per_epoch
    
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            output_dict = model(
                **input_dict,
                use_text_emb_in_suffix_sam=args.use_text_emb_in_suffix_sam,
            )

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_loss = output_dict["mask_loss"]
            kl_loss = output_dict["kl_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            kl_losses.update(kl_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_losses.all_reduce()
                kl_losses.all_reduce()

            if args.local_rank == 0:
                # 计算时间预测
                current_step = global_step + 1
                current_global_step_total = current_global_step + current_step
                
                # 当前epoch剩余时间
                remaining_steps_in_epoch = args.steps_per_epoch - current_step
                eta_epoch = remaining_steps_in_epoch * batch_time.avg
                
                # 总训练剩余时间
                remaining_total_steps = total_steps - current_global_step_total
                eta_total = remaining_total_steps * batch_time.avg
                
                # 训练进度百分比
                progress_percent = (current_global_step_total / total_steps) * 100
                
                # 显示时间预测
                time_info = f"ETA Epoch: {format_time(eta_epoch)} | ETA Total: {format_time(eta_total)} | Progress: {progress_percent:.1f}%"
                logger.info(f"Epoch [{epoch}][{current_step}/{args.steps_per_epoch}] {time_info}")
                
                progress.display(global_step + 1)
                writer.add_scalar(
                    "train/loss", losses.avg, global_step + epoch * args.steps_per_epoch
                )
                writer.add_scalar(
                    "train/ce_loss",
                    ce_losses.avg,
                    global_step + epoch * args.steps_per_epoch,
                )
                writer.add_scalar(
                    "train/mask_loss",
                    mask_losses.avg,
                    global_step + epoch * args.steps_per_epoch,
                )
                writer.add_scalar(
                    "train/kl_loss",
                    kl_losses.avg,
                    global_step + epoch * args.steps_per_epoch,
                )
                writer.add_scalar(
                    "metrics/total_secs_per_batch",
                    batch_time.avg,
                    global_step + epoch * args.steps_per_epoch,
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch",
                    data_time.avg,
                    global_step + epoch * args.steps_per_epoch,
                )
                writer.add_scalar(
                    "metrics/eta_epoch_seconds",
                    eta_epoch,
                    global_step + epoch * args.steps_per_epoch,
                )
                writer.add_scalar(
                    "metrics/eta_total_seconds",
                    eta_total,
                    global_step + epoch * args.steps_per_epoch,
                )
                writer.add_scalar(
                    "metrics/progress_percent",
                    progress_percent,
                    global_step + epoch * args.steps_per_epoch,
                )
                
                # 记录到Swanlab
                swanlab.log({
                    "train/loss": losses.avg,
                    "train/ce_loss": ce_losses.avg,
                    "train/mask_loss": mask_losses.avg,
                    "train/kl_loss": kl_losses.avg,
                    "metrics/total_secs_per_batch": batch_time.avg,
                    "metrics/data_secs_per_batch": data_time.avg,
                    "metrics/eta_epoch_seconds": eta_epoch,
                    "metrics/eta_total_seconds": eta_total,
                    "metrics/progress_percent": progress_percent,
                    "epoch": epoch,
                    "step": global_step + epoch * args.steps_per_epoch,
                })

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_losses.reset()
            kl_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar(
                    "train/lr", curr_lr[0], global_step + epoch * args.steps_per_epoch
                )
                swanlab.log({"train/lr": curr_lr[0]})

    # 记录epoch完成时间
    if args.local_rank == 0:
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch [{epoch}] completed in {format_time(epoch_time)}")

    return train_iter


if __name__ == "__main__":
    main(sys.argv[1:]) 