import os
import sys
import warnings

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config.tiny_config_isic import setting_config
from datasets.dataset import NPY_datasets
from engine import train_one_epoch, val_one_epoch
from models.wfdvim import WFDViM
from utils import get_logger, get_optimizer, get_scheduler, log_config_info, set_seed

warnings.filterwarnings("ignore")

def main(config):
    sys.path.append(config.work_dir + "/")
    log_dir = os.path.join(config.work_dir, "log")
    checkpoint_dir = os.path.join(config.work_dir, "checkpoints")
    resume_model = os.path.join(checkpoint_dir, "latest.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(config.work_dir, "outputs"), exist_ok=True)

    logger = get_logger("train", log_dir)
    writer = SummaryWriter(os.path.join(config.work_dir, "summary"))
    log_config_info(config, logger)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    train_dataset = NPY_datasets(config.data_path, config, train=True)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False,
    )

    model = WFDViM().cuda()
    criterion = config.criterion.cuda()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    start_epoch = 1
    best_metrics = {
        "dice": {"value": 0.0, "epoch": 0},
        "miou": {"value": 0.0, "epoch": 0},
        "accuracy": {"value": 0.0, "epoch": 0},
        "assd": {"value": float("inf"), "epoch": 0},
    }

    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device("cpu"))
        model.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metrics["dice"]["value"] = checkpoint.get("best_dice", 0.0)
        best_metrics["dice"]["epoch"] = checkpoint.get("best_epoch", 0)
        logger.info(f"Resumed from {resume_model} at epoch {start_epoch}")

    step = 0
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
        )

        if epoch % config.val_interval == 0:
            val_loss, dice, miou, accuracy, assd_score = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config,
            )
            if dice > best_metrics["dice"]["value"]:
                best_metrics["dice"] = {"value": dice, "epoch": epoch}
                torch.save(model.model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            if miou > best_metrics["miou"]["value"]:
                best_metrics["miou"] = {"value": miou, "epoch": epoch}
            if accuracy > best_metrics["accuracy"]["value"]:
                best_metrics["accuracy"] = {"value": accuracy, "epoch": epoch}
            if assd_score < best_metrics["assd"]["value"]:
                best_metrics["assd"] = {"value": assd_score, "epoch": epoch}

        torch.save(
            {
                "epoch": epoch,
                "best_dice": best_metrics["dice"]["value"],
                "best_epoch": best_metrics["dice"]["epoch"],
                "model_state_dict": model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, "latest.pth"),
        )

    logger.info(f"Best Dice: {best_metrics['dice']['value']:.4f} at epoch {best_metrics['dice']['epoch']}")
    logger.info(f"Best mIoU: {best_metrics['miou']['value']:.4f} at epoch {best_metrics['miou']['epoch']}")
    logger.info(f"Best Accuracy: {best_metrics['accuracy']['value']:.4f} at epoch {best_metrics['accuracy']['epoch']}")
    logger.info(f"Best ASSD: {best_metrics['assd']['value']:.4f} at epoch {best_metrics['assd']['epoch']}")

if __name__ == "__main__":
    main(setting_config)
