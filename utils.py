import logging
import logging.handlers
import math
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from matplotlib import pyplot as plt
from thop import profile

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    info_name = os.path.join(log_dir, f"{name}.info.log")
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name, when="D", encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    return logger

def log_config_info(config, logger):
    logger.info("Config info")
    for key, value in config.__dict__.items():
        if not key.startswith("_"):
            logger.info(f"{key}: {value}")

def get_optimizer(config, model):
    if config.opt == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    if config.opt == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    if config.opt == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {config.opt}")

def get_scheduler(config, optimizer):
    if config.sch == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch,
        )
    if config.sch == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch,
        )
    if config.sch == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch,
        )
    if config.sch == "WP_CosineLR":
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
            math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    raise ValueError(f"Unsupported scheduler: {config.sch}")

def save_imgs(img, mask, pred, index, save_path, datasets, threshold=0.5, test_data_name=None):
    os.makedirs(save_path, exist_ok=True)
    image = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image = image / 255.0 if image.max() > 1.1 else image
    mask = np.where(np.squeeze(mask, axis=0) > 0.5, 1, 0)
    pred = np.where(np.squeeze(pred, axis=0) > threshold, 1, 0)
    figure = plt.figure(figsize=(10, 15))
    plt.subplot(3, 1, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(3, 1, 2)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.subplot(3, 1, 3)
    plt.imshow(pred, cmap="gray")
    plt.axis("off")
    prefix = f"{test_data_name}_" if test_data_name is not None else ""
    figure.savefig(os.path.join(save_path, f"{prefix}{index}.png"), bbox_inches="tight", pad_inches=0)
    plt.close(figure)

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        return self.bceloss(pred.view(size, -1), target.view(size, -1))

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1.0
        size = pred.size(0)
        pred = pred.view(size, -1)
        target = target.view(size, -1)
        intersection = pred * target
        score = (2 * intersection.sum(1) + smooth) / (pred.sum(1) + target.sum(1) + smooth)
        return 1 - score.sum() / size

class BceDiceLoss(nn.Module):
    def __init__(self, wd=1, wb=1):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        return self.wd * self.dice(pred, target) + self.wb * self.bce(pred, target)

class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss=None):
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else BceDiceLoss()

    def forward(self, preds, target):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        loss = 0.0
        for pred in preds:
            if pred.shape[2:] != target.shape[2:]:
                pred = torch.nn.functional.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=True)
            loss = loss + self.base_loss(torch.sigmoid(pred), target)
        return loss

def cal_params_flops(model, size, logger=None):
    model = model.cuda().eval()
    input_tensor = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    total = sum(p.numel() for p in model.parameters())
    message = f"flops: {flops / 1e9:.6f} G, params: {params / 1e6:.6f} M, total: {total / 1e6:.6f} M"
    print(message)
    if logger is not None:
        logger.info(message)
    return flops, params
