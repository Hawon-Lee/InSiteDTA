import os, sys
import json
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F


def add_gaussian_noise(labels, noise_std):
    noise = torch.normal(mean=0.0, std=noise_std, size=labels.shape).to(labels.device)
    noisy_labels = labels + noise

    noisy_labels = torch.clamp(noisy_labels, min=0)

    return noisy_labels


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, trace_func=print):
        """
        Early stopping implementation

        Args:
            patience (int): Number of epochs to wait after validation loss stops improving
            min_delta (float): Minimum change to qualify as an improvement
            trace_func (function): Function to use for printing messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.trace_func = trace_func
        self.best_epoch = 0
        self.current_epoch = 0

    def __call__(self, val_loss, epoch=None):
        """
        Determine whether to stop early based on validation loss.

        Args:
            val_loss (float): Current validation loss value
            epoch (int, optional): Current epoch number
        """
        if epoch is not None:
            self.current_epoch = epoch

        # Check for improved validation loss - uses same condition as train_model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = self.current_epoch
            return True  # Return that improvement occurred
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.trace_func(
                    f"Early stopping triggered! Best epoch: {self.best_epoch}"
                )
            return False  # Return that no improvement occurred


def rotate_3d_6faces(voxel, pocket):
    """
    Randomly rotate 3D data so that one of the six faces of the cube faces forward.

    Args:
        voxel (torch.Tensor): Input voxel data (B, C, D, H, W)
        pocket (torch.Tensor): Target pocket data (B, 1, D, H, W)

    Returns:
        tuple: Rotated (voxel, pocket) pair
    """
    # Define transformations for 6 directions
    # Each transformation is a simple axis transformation that makes a specific face face forward
    faces_rotations = [
        lambda x: x,  # Front face (no change)
        lambda x: x.flip(4),  # Back face
        lambda x: x.transpose(2, 4),  # Left face
        lambda x: x.transpose(2, 4).flip(4),  # Right face
        lambda x: x.transpose(3, 4),  # Top face
        lambda x: x.transpose(3, 4).flip(4),  # Bottom face
    ]

    # Randomly select a rotation
    rotation = random.choice(faces_rotations)

    # Apply the same rotation to both voxel and pocket
    rotated_voxel = rotation(voxel)
    rotated_pocket = rotation(pocket)

    return rotated_voxel, rotated_pocket


def fix_seed(
    seed: int = 312, deterministic: bool = True, benchmark: bool = False
) -> None:
    """
    Fix all random seeds for reproducibility

    Args:
        seed (int): Seed number to use. Default is 312.
        deterministic (bool): If True, ensures deterministic behavior in CuDNN backend.
                            May impact performance. Default is True.

        benchmark (bool): If True, enables cudnn.benchmark for potentially faster training.
                         Only set to True if input sizes don't change. Default is False.
    """

    # Python random seed
    random.seed(seed)

    # Numpy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)

    # CUDA random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

        # CUDA backend settings
        cudnn.deterministic = deterministic  # Set to True for perfect reproducibility
        cudnn.benchmark = benchmark  # Set to True for better performance with consistent input sizes

        torch.use_deterministic_algorithms(deterministic)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten predictions and targets
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()

        # Calculate Dice coefficient and loss
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class DiceWithLogitsLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceWithLogitsLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Flatten preds and targets
        B = logits.shape[0]
        probs = torch.sigmoid(logits)
        probs = probs.reshape(B, -1)
        targets = targets.reshape(B, -1)

        # Calculate intersection and union
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        # Calculate Dice coefficient and loss
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class SoftDiceWithLogitsLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftDiceWithLogitsLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Compute Soft Dice Loss for binary classification

        Args:
            predictions (torch.Tensor): Model output logits [B, 1, *] or [B, *]
            targets (torch.Tensor): Ground truth labels [B, 1, *] or [B, *]

        Returns:
            torch.Tensor: Soft Dice Loss value
        """
        # Add channel dimension if not present
        if predictions.dim() == targets.dim() and predictions.size(1) != 1:
            predictions = predictions.unsqueeze(1)
            targets = targets.unsqueeze(1)

        # Get batch size
        batch_size = predictions.size(0)

        # Flatten predictions and targets (keeping batch dimension)
        predictions = torch.sigmoid(predictions)
        predictions = predictions.reshape(batch_size, -1)  # Convert to [B, I] shape
        targets = targets.reshape(batch_size, -1)  # Convert to [B, I] shape

        # Calculate numerator: sum of element-wise product of predictions and targets
        intersection = (predictions * targets).sum(dim=1)  # [B]

        # Calculate denominator: sum of squared predictions + sum of squared targets
        denominator = (predictions**2).sum(dim=1) + (targets**2).sum(dim=1)  # [B]

        # Calculate Dice coefficient for each batch
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)  # [B]

        # Average across batch and return loss
        dice_loss = 1 - dice.mean()

        return dice_loss


def calc_DCC_with_logit(pred, true, voxel_size, threshold):
    """
    Calculate voxel-based DCC(Distance between Center to Center)
    Centers are calculated by averaging the center coordinates of each binding site voxel per batch (No atom-type consideration)
    true -> (batch, 1, n_voxels, n_voxels, n_voxels)
    pred -> (batch, 1, n_voxels, n_voxels, n_voxels)
    voxel_size -> length of one side of a voxel
    threshold -> voxels higher than this value are designated as binding sites for metric calculation
    """
    with torch.no_grad():
        pred = torch.sigmoid(pred)
        pred_binary = pred > threshold
        batch_size = pred.shape[0]
        nan_index = []

        pred_centers_list = []
        true_centers_list = []

        for b in range(batch_size):
            pred_coords = torch.where(pred_binary[b, 0])
            true_coords = torch.where(true[b, 0])

            if len(pred_coords[0]) == 0:
                nan_index.append(b)
                continue

            pred_center = torch.stack(pred_coords).float().mean(dim=1) * voxel_size
            true_center = torch.stack(true_coords).float().mean(dim=1) * voxel_size

            if torch.isnan(pred_center).any():
                nan_index.append(b)
                continue

            pred_centers_list.append(pred_center)
            true_centers_list.append(true_center)

            del pred_coords, true_coords

        if not pred_centers_list:
            return torch.tensor([0]).to(pred.device), nan_index

        # Convert batch-wise center points to tensors
        pred_centers = torch.stack(pred_centers_list)
        true_centers = torch.stack(true_centers_list)

        # Calculate batch-wise distances
        dists = torch.norm(pred_centers - true_centers, dim=1)

        # Explicitly release intermediate variables
        del pred_binary, pred_centers_list, true_centers_list

        return dists, nan_index


def calc_DVO_with_logit(pred, true, threshold):
    """
    Calculate voxel-based DVO (Discretized Volume Overlap)
    Volume intersection / union
    pred -> (batch, 1, n_voxels, n_voxels, n_voxels) - model output logits
    true -> (batch, 1, n_voxels, n_voxels, n_voxels) - ground truth binary labels
    threshold -> voxels above this value are designated as binding sites for metric calculation
    """
    # Apply sigmoid and thresholding
    pred = torch.sigmoid(pred)
    pred = pred > threshold
    intersection = (pred * true > 0).sum(dim=[1, 2, 3, 4])
    union = (pred + true > 0).sum(dim=[1, 2, 3, 4])
    DVOs = intersection / union

    return DVOs


from sklearn.metrics import f1_score


def calc_f1_score(pred, true):
    pred_binary = (pred > 0.5).flatten().cpu().numpy()
    target_binary = true.flatten().cpu().numpy()
    f1 = f1_score(target_binary, pred_binary)

    return f1


def calc_f1_score_logit(logit, true):
    pred = torch.sigmoid(logit)
    pred_binary = (pred > 0.5).flatten().cpu().numpy()
    target_binary = true.flatten().cpu().numpy()
    f1 = f1_score(target_binary, pred_binary)

    return f1


import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class ExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


def parse_int_list(value):
    if isinstance(value, list):
        return value
    # process multiple arguments like "[1, 1, 1]"
    if isinstance(value, str) and value.startswith("["):
        return [int(x.strip()) for x in value.strip("[]").split(",")]
    # process multiple arguments like "1 1 1"
    return [int(x) for x in value.split()]


def parse_float_list(value):
    if isinstance(value, list):
        return value
    # process multiple arguments like "[1, 1, 1]"
    if isinstance(value, str) and value.startswith("["):
        return [float(x.strip()) for x in value.strip("[]").split(",")]
    # process multiple arguments like "1 1 1"
    return [float(x) for x in value.split()]


def parse_str_list(value):
    if isinstance(value, list):
        return value
    # process multiple arguments like "[1, 1, 1]"
    if isinstance(value, str) and value.startswith("["):
        return [str(x.strip()) for x in value.strip("[]").split(",")]
    # process multiple arguments like "1 1 1"
    return [str(x) for x in value.split()]


def override_args_from_json(args, wandb_json_path):
    """
    Override args with values from JSON config file.
    Command-line arguments take precedence over JSON config.

    Args:
        args: Parsed arguments from argparse
        json_path: Path to JSON config file (wandb config style with nested values)

    Returns:
        args: Updated arguments
    """
    if wandb_json_path is None:
        return args

    # Load JSON config
    with open(wandb_json_path, "r") as f:
        config_dict = json.load(f)

    # Detect explicitly provided command-line arguments
    cli_provided_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            arg_name = arg[2:].split("=")[0]
            cli_provided_args.add(arg_name)

    # Update args with JSON values (skip explicitly provided ones)
    for key, value in config_dict.items():
        if key in ["_wandb", "config"]:  # Skip internal keys
            continue

        # Handle nested dict structure
        if isinstance(value, dict) and "value" in value:
            value = value["value"]

        # Apply JSON value only if not explicitly set via CLI
        if hasattr(args, key) and key not in cli_provided_args:
            setattr(args, key, value)

    return args
