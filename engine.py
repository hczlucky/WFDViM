import numpy as np
import torch
from medpy.metric import assd
from medpy.metric.binary import hd95
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer):
    model.train()
    loss_list = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
    pbar.set_description(f"Train Epoch {epoch}")

    for iteration, data in pbar:
        step += iteration
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        writer.add_scalar("loss", loss, global_step=step)
        pbar.set_postfix(loss=f"{np.mean(loss_list):.4f}", lr=f"{lr:.6f}")
        if iteration % config.print_interval == 0:
            logger.info(f"train: epoch {epoch}, iter:{iteration}, loss: {np.mean(loss_list):.4f}, lr: {lr}")

    scheduler.step()
    return step

def calculate_assd(pred, target):
    try:
        pred = np.asarray(pred).squeeze()
        target = np.asarray(target).squeeze()
        pred = (pred > 0.5).astype(np.uint8)
        target = (target > 0.5).astype(np.uint8)
        if not np.any(pred) and not np.any(target):
            return 0.0
        if not np.any(pred) or not np.any(target):
            return 100.0
        value = assd(pred, target)
        if np.isnan(value) or np.isinf(value):
            return 100.0
        return float(value)
    except Exception:
        return 100.0

def compute_hd95_2d(pred, target, threshold=0.5):
    pred = np.asarray(pred).squeeze()
    target = np.asarray(target).squeeze()
    pred = pred >= threshold
    target = target >= 0.5
    if not np.any(pred) and not np.any(target):
        return 0.0
    if not np.any(pred) or not np.any(target):
        return 100.0
    return float(hd95(pred, target, voxelspacing=None))

def _binary_metrics(preds, targets, threshold):
    preds_flat = np.array(preds).reshape(-1)
    targets_flat = np.array(targets).reshape(-1)
    y_pred = np.where(preds_flat >= threshold, 1, 0)
    y_true = np.where(targets_flat >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    accuracy = float(tn + tp) / float(np.sum(confusion)) if np.sum(confusion) else 0.0
    sensitivity = float(tp) / float(tp + fn) if tp + fn else 0.0
    specificity = float(tn) / float(tn + fp) if tn + fp else 0.0
    dice = float(2 * tp) / float(2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    miou = float(tp) / float(tp + fp + fn) if tp + fp + fn else 0.0
    return dice, miou, accuracy, sensitivity, specificity, confusion

def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    model.eval()
    preds = []
    targets = []
    preds_2d = []
    targets_2d = []
    losses = []

    with torch.no_grad():
        pbar = tqdm(test_loader, total=len(test_loader))
        pbar.set_description(f"Val Epoch {epoch}")
        for data in pbar:
            image, mask = data
            image = image.cuda(non_blocking=True).float()
            mask = mask.cuda(non_blocking=True).float()
            output = model(image)
            loss = criterion(output, mask)
            losses.append(loss.item())
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = torch.sigmoid(output)
            current_mask = mask.squeeze(1).cpu().numpy()
            current_output = output.squeeze(1).cpu().numpy()
            targets.append(current_mask)
            preds.append(current_output)
            for i in range(current_mask.shape[0]):
                targets_2d.append(current_mask[i])
                preds_2d.append(current_output[i])
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

    dice, miou, accuracy, sensitivity, specificity, confusion = _binary_metrics(preds, targets, config.threshold)
    assd_scores = [calculate_assd(pred, target) for pred, target in zip(preds_2d, targets_2d)]
    hd95_scores = [compute_hd95_2d(pred, target, config.threshold) for pred, target in zip(preds_2d, targets_2d)]
    valid_assd = [score for score in assd_scores if score < 50.0]
    assd_score = float(np.mean(valid_assd)) if valid_assd else 100.0
    hd95_score = float(np.mean(hd95_scores)) if hd95_scores else 100.0
    log_info = {
        "epoch": epoch,
        "loss": float(np.mean(losses)),
        "dice": dice,
        "miou": miou,
        "accuracy": accuracy,
        "assd": assd_score,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "hd95": hd95_score,
        "confusion_matrix": confusion.tolist(),
    }
    print(log_info)
    logger.info(str(log_info))
    return float(np.mean(losses)), dice, miou, accuracy, assd_score
