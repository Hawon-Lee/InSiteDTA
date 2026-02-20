# TODO: 저장을 ckpt 폴더에 맞게 해서 inference 폴더 또는 reproduce 폴더와 일치하게.
"""
data directory, hyperparams -> training cfg file
training cfg file -> training/validation -> ckpt saving
evaluation by inference code? options 조절이 안되는데 out channels 만 조절하면 되려나
"""
import argparse
import os
import wandb
import json
import numpy as np

from typing import Literal, Optional
from functools import partial
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from src.scripts.dataloader import MasterDataLoader
from src.scripts.model.model import InSiteDTA
from src.scripts.utils import print_args, parse_to_list
from src.scripts.utils_train import (
    fix_seed,
    CosineAnnealingWarmUpRestarts,
    rotate_3d_6faces,
    add_gaussian_noise,
    EarlyStopping,
    CosineAnnealingWarmUpRestarts,
    DiceWithLogitsLoss,
    SoftDiceWithLogitsLoss,
    calc_DCC_with_logit,
    calc_DVO_with_logit,
)

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    parser.add_argument(
        "--wandb_config",
        type=str,
        help="Path to result JSON config file after wandb sweep",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="Json configuration file path specifies train/test data splits and metadata for learning.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Output directory to save best model validation results and checkpoint",
    )
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of dataloader workers"
    )

    # For Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=312,
        help="Random seed for training, (-1 for random)",
    )

    # =============== Model Args (sw: swin transformer encoder, mol: molecule encoder, dec: decoder, ca: cross attention module) ===============
    parser.add_argument(
        "--in_channels", type=int, default=21, help="Input channel size"
    )
    parser.add_argument(
        "--out_channels", type=int, default=1, help="Output channel size"
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=48,
        help="Dimension of the first embedding token",
    )

    parser.add_argument(
        "--sw_depths",
        type=partial(parse_to_list, type=int),
        default=[2, 2, 2],
        help="Swin Transformer layer depths. Each int means number of swin attention operations in each layer",
    )
    parser.add_argument(
        "--sw_window_size",
        type=int,
        default=7,
        help="Window size for swin attention operation",
    )
    parser.add_argument(
        "--sw_patch_size",
        type=partial(parse_to_list, type=int),
        default=[2, 2, 2],
        help="Patch size (only used when model is SwinUnetr)",
    )
    parser.add_argument(
        "--sw_num_heads",
        type=partial(parse_to_list, type=int),
        default=[3, 6, 12],
        help="Head numbers for multi-head attention in each Swin Transformer layer",
    )
    parser.add_argument(
        "--sw_mlp_ratio",
        type=float,
        default=4.0,
        help="MLP hidden size ratio for feedforward networks in Swin Transformer",
    )
    parser.add_argument(
        "--sw_qkv_bias",
        type=bool,
        default=True,
        help="Whether to use bias in query, key, value projections in Swin Transformer attention",
    )
    parser.add_argument(
        "--sw_drop_rate",
        type=float,
        default=0.0,
        help="Dropout rate for feedforward networks in Swin Transformer",
    )
    parser.add_argument(
        "--sw_attn_drop_rate",
        type=float,
        default=0.0,
        help="Dropout rate for swin attention operations in Swin Transformer",
    )
    parser.add_argument(
        "--sw_drop_path_rate",
        type=float,
        default=0.1,
        help="Drop path rate for stochastic depth in Swin Transformer",
    )
    parser.add_argument(
        "--sw_act",
        type=str,
        default="gelu",
        help="Activation function type for Swin Transformer",
    )

    parser.add_argument(
        "--mol_encoder_types",
        type=str,
        default=["attnfp", "schnet", "egnn"],
        help="Molecular graph encoder combinations (choices: attnfp, schnet, egnn, gcn, gat)",
    )
    parser.add_argument("--mol_in_channels", type=int, default=54)
    parser.add_argument("--mol_hidden_channels", type=int, default=128)
    parser.add_argument("--mol_out_channels", type=int, default=128)
    parser.add_argument("--mol_num_layers", type=int, default=2)
    parser.add_argument("--mol_num_interactions_3d", type=int, default=3)
    parser.add_argument("--mol_dropout_3d", type=float, default=0.15)
    parser.add_argument("--mol_cutoff_3d", type=int, default=10)
    parser.add_argument("--mol_num_filters_schnet", type=int, default=128)
    parser.add_argument("--mol_edge_num_gaussian_schnet", type=int, default=50)
    parser.add_argument("--mol_edge_num_fourier_feats_egnn", type=int, default=3)
    parser.add_argument("--mol_soft_edge_egnn", type=bool, default=False)
    parser.add_argument("--mol_act", type=str, default="mish")

    parser.add_argument("--dec_drop_rate", type=float, default=0.1)
    parser.add_argument("--dec_act", type=str, default="gelu")

    parser.add_argument("--ca_num_heads", type=int, default=8)
    parser.add_argument("--ca_dropout", type=float, default=0.1)

    # =============== Training Args ===============
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--tr_subset_ratio",
        type=float,
        default=1.0,
        help="Ratio of training data to use per epoch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer (=basal level of cosine scheduler)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=48, help="Batch size for training"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps to accumulate gradients before performing a backward/update pass",
    )

    # loss configs
    parser.add_argument(
        "--pocket_loss_weight",
        type=float,
        default=2.0,
        help="Loss weight for pocket prediction",
    )
    parser.add_argument(
        "--aff_loss_weight",
        type=float,
        default=0.2,
        help="Loss weight for affinity prediction",
    )
    parser.add_argument(
        "--aff_loss_type",
        type=str,
        default="mse",
        choices=["mse", "smooth_l1", "huber"],
        help="Affinity prediction loss function type",
    )
    parser.add_argument(
        "--pocket_loss_types",
        type=partial(parse_to_list, type=str),
        default="soft_dice bce",
        help="Muitiple pocket loss types to use, seperated with single space (e.g., 'soft_dice dice bce')",
    )

    # optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam"],
        help="Optimizer types",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for Adam(w) optimizer",
    )

    # scheduler
    parser.add_argument(
        "--scheduler_T0",
        type=int,
        default=100,
        help="First cycle epoch count for CosineAnnealingWarmUpRestarts",
    )
    parser.add_argument(
        "--scheduler_T_mult",
        type=int,
        default=1,
        help="Cycle length multiplier for CosineAnnealingWarmUpRestarts",
    )
    parser.add_argument(
        "--scheduler_eta_max",
        type=float,
        default=1e-3,
        help="Maximum learning rate for CosineAnnealingWarmUpRestarts",
    )
    parser.add_argument(
        "--scheduler_T_up",
        type=int,
        default=10,
        help="Warmup epoch count for CosineAnnealingWarmUpRestarts",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.7,
        help="Decay factor per cycle for CosineAnnealingWarmUpRestarts",
    )

    # augmentation
    parser.add_argument(
        "--rotation_prob",
        type=float,
        default=0.3,
        help="Probability of applying random rotation",
    )
    parser.add_argument(
        "--label_noise_std",
        type=float,
        default=0.15,
        help="Std value for label noise injection",
    )

    # earlystop
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )

    # =============== Evaluation Args ===============
    parser.add_argument(
        "--DCC_threshold",
        type=float,
        default=0.5,
        help="Threshold to classify voxel as pocket (DCC)",
    )
    parser.add_argument(
        "--DVO_threshold",
        type=float,
        default=0.5,
        help="Threshold to classify voxel as pocket (DVO)",
    )
    parser.add_argument(
        "--DCC_SR_threshold",
        type=float,
        default=4.0,
        help="Max DCC value for success rate calculation in validation",
    )

    args = parser.parse_args()
    return args


def get_configs() -> tuple[dict, dict]:
    train_cfg = get_arguments()
    print_args(train_cfg)

    with open(train_cfg.data_config, "r") as fp:
        data_cfg = json.load(fp)

    return data_cfg, vars(train_cfg)


def train_model(
    model: nn.Module,
    tr_loader: torch.utils.data.DataLoader,
    vl_loader: torch.utils.data.DataLoader,
    train_cfg: dict,
    voxel_size: int,
    exp_name: str,
):

    wandb.init(
        project="InSiteDTA",
        name=exp_name,
        config=train_cfg,
    )
    wandb.watch(model, log="gradients", log_freq=50)
    
    aug_generator = torch.Generator()
    if train_cfg["seed"] != -1:
        aug_generator.manual_seed(train_cfg["seed"])

    device = torch.device(f"cuda:{train_cfg['gpu']}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    early_stopping = EarlyStopping(patience=train_cfg["patience"])    
    scaler = GradScaler()
    
    if train_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
    elif train_cfg["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=train_cfg["scheduler_T0"],
        T_mult=train_cfg["scheduler_T_mult"],
        eta_max=train_cfg["scheduler_eta_max"],
        T_up=train_cfg["scheduler_T_up"],
        gamma=train_cfg["scheduler_gamma"],
    )


    aff_criterion = nn.MSELoss()
    poc_criterion_bce = nn.BCEWithLogitsLoss()
    poc_criterion_softdice = SoftDiceWithLogitsLoss()

    poc_weight = train_cfg["pocket_loss_weight"]
    aff_weight = train_cfg["aff_loss_weight"]

    best_vl_loss = float("inf")
    best_model = None
    nan_count_ls = []

    tr_dataset_size = len(tr_loader.dataset)
    vl_dataset_size = len(vl_loader.dataset)

    tr_steps_per_epoch = len(tr_loader)

    if train_cfg["tr_subset_ratio"] < 1.0:
        tr_iter_limit = int(tr_steps_per_epoch * train_cfg["tr_subset_ratio"])
        tr_iter_limit = max(1, tr_iter_limit)
        print(f"Speed-up: Using {train_cfg['tr_subset_ratio']*100}% data (Train: {tr_iter_limit}/{tr_steps_per_epoch}, Validation on full set")
    else:
        tr_iter_limit = tr_steps_per_epoch

    # TRAINING
    for epoch in range(train_cfg["epochs"]):
        # Training phase
        model.train()
        tr_recon_losses = 0
        tr_pocket_losses = 0
        tr_aff_losses = 0
        tr_total_losses = 0
        accumulation_step = 0

        for i, sample in enumerate(tqdm(tr_loader)):
            if i >= tr_iter_limit:
                break
            voxel, pocket, ligand_data, b_aff = (
                sample["voxel"],
                sample["pocket_label"],
                sample["ligand_data"],
                sample["b_aff"],
            )
            if pocket.sum() == 0.0:
                print(sample['data_key'], "has no pocket.")
            voxel, pocket, ligand_data, b_aff = (
                voxel.to(device),
                pocket.to(device),
                ligand_data.to(device),
                b_aff.to(device),
            )
            
            # On-line data augmentation
            if (
                torch.rand(1, generator=aug_generator).item()
                < train_cfg["rotation_prob"]
            ):
                voxel, pocket = rotate_3d_6faces(voxel, pocket)

            b_aff = add_gaussian_noise(b_aff, noise_std=train_cfg["label_noise_std"])

            # Zero gradients only at the beginning of accumulation
            if accumulation_step == 0:
                optimizer.zero_grad()

            with autocast(device_type=device.type):
                output_voxel, output_aff = model(voxel, ligand_data)
                recon_voxel = output_voxel[:, :train_cfg["in_channels"], ...]
                pred_pocket = output_voxel[:, -1:, ...]  # It's a logit.

                # Reconstruction loss for first 11 channels
                batch_size = voxel.size(0)
                recon_loss = torch.tensor([0]).to(device)
                # recon_loss *= recon_weight

                pocket_loss = 0
                for loss_type, loss_fn in pocket_loss_functions.items():
                    pocket_loss += loss_fn(pred_pocket, pocket)

                # Binding affinity loss
                has_valid_aff = ~torch.isnan(b_aff).all()
                if not has_valid_aff:
                    breakpoint()
                aff_loss = aff_criterion(output_aff, b_aff) if has_valid_aff else torch.tensor(0.0, device=voxel.device)
                # Total loss
                total_loss = (
                    recon_weight * recon_loss
                    + poc_weight * pocket_loss
                    + aff_weight * aff_loss
                )
                
                # Normalize loss for gradient accumulation
                total_loss = total_loss / train_cfg["grad_accumulation_steps"]

            scaler.scale(total_loss).backward()
            accumulation_step += 1
            
            # Perform optimizer step only when accumulation is complete or at the end of epoch
            if accumulation_step == train_cfg["grad_accumulation_steps"] or i == len(tr_loader) - 1:
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                accumulation_step = 0  # Reset accumulation counter

            # Store losses (multiply total_loss back by accumulation steps for proper logging)
            tr_recon_losses += recon_loss.detach().item()
            tr_pocket_losses += pocket_loss.detach().item()
            tr_aff_losses += aff_loss.detach().item()
            tr_total_losses += total_loss.detach().item() * train_cfg["grad_accumulation_steps"]

            # del voxel, pocket, ligand_data, b_aff
            # del output_voxel, output_aff, recon_voxel, pred_pocket
            # del recon_loss, pocket_dice_loss, pocket_bce_loss, pocket_loss, aff_loss, total_loss

            if i % 20 == 0:
                torch.cuda.empty_cache()

        avg_tr_recon_loss = tr_recon_losses / tr_iter_limit
        avg_tr_pocket_loss = tr_pocket_losses / tr_iter_limit
        avg_tr_aff_loss = tr_aff_losses / tr_iter_limit
        avg_tr_total_loss = tr_total_losses / tr_iter_limit

        # Validation phase
        model.eval()
        vl_recon_losses = []
        vl_pocket_losses = []
        vl_aff_losses = []
        vl_total_losses = []
        vl_vDCCs = []  # voxel-base calculated DCC (not per-atomtype)
        vl_DVOs = []
        vl_nan_indice = []
        vl_aff_all_preds = []
        vl_aff_all_targets = []

        # VALIDATION
        with torch.no_grad():
            for sample in tqdm(vl_loader):
                voxel, pocket, ligand_data, b_aff = (
                    sample["voxel"],
                    sample["pocket_label"],
                    sample["ligand_data"],
                    sample["b_aff"],
                )
                voxel, pocket, ligand_data, b_aff = (
                    voxel.to(device),
                    pocket.to(device),
                    ligand_data.to(device),
                    b_aff.to(device),
                )
                
                if pocket.sum() == 0.0:
                    print(sample['data_key'])

                with autocast(device_type=device.type):
                    output_voxel, output_aff = model(voxel, ligand_data)
                    recon_voxel = output_voxel[:, :train_cfg["in_channels"], ...]
                    pred_pocket = output_voxel[:, -1:, ...]

                    batch_size = voxel.size(0)
                    recon_loss = torch.tensor([0]).to(device)

                    pocket_loss = 0
                    for loss_type, loss_fn in pocket_loss_functions.items():
                        pocket_loss += loss_fn(pred_pocket, pocket) * batch_size
                    
                    has_valid_aff = ~torch.isnan(b_aff).all()
                    aff_loss = aff_criterion(output_aff, b_aff) * batch_size if has_valid_aff else torch.tensor(0.0, device=voxel.device)

                    total_loss = (
                        recon_weight * recon_loss
                        + poc_weight * pocket_loss
                        + aff_weight * aff_loss
                    )

                # val metrics 계산
                vDCC, nan_index = calc_DCC_with_logit(
                    pred_pocket,
                    pocket,
                    voxel_size=voxel_size,
                    threshold=train_cfg["DCC_threshold"],
                )
                    
                DVO = calc_DVO_with_logit(
                    pred_pocket, pocket, threshold=train_cfg["DVO_threshold"]
                )

                vl_recon_losses.append(recon_loss.item())
                vl_pocket_losses.append(pocket_loss.item())
                vl_aff_losses.append(aff_loss.item())
                vl_total_losses.append(total_loss.item())
                vl_vDCCs += vDCC.tolist()
                vl_DVOs += DVO.tolist()
                vl_nan_indice += nan_index
                vl_aff_all_preds.append(output_aff.detach().cpu())
                vl_aff_all_targets.append(b_aff.cpu())

        avg_vl_recon_loss = sum(vl_recon_losses) / vl_dataset_size
        avg_vl_pocket_loss = sum(vl_pocket_losses) / vl_dataset_size
        avg_vl_aff_loss = sum(vl_aff_losses) / vl_dataset_size
        avg_vl_total_loss = sum(vl_total_losses) / vl_dataset_size
        avg_vl_vDCC = sum(vl_vDCCs) / vl_dataset_size
        avg_vl_vDCC_SR = (
            len([DCC for DCC in vl_vDCCs if DCC <= train_cfg["DCC_SR_threshold"]])
            / vl_dataset_size
        )
        avg_vl_DVO = sum(vl_DVOs) / vl_dataset_size
        vl_DCC_nan_count = len(vl_nan_indice)
        nan_count_ls.append(vl_DCC_nan_count)

        epoch_preds = torch.cat(vl_aff_all_preds, dim=0).cpu().numpy().squeeze()
        epoch_targets = torch.cat(vl_aff_all_targets, dim=0).cpu().numpy().squeeze()
        vl_PCC = np.corrcoef(epoch_preds, epoch_targets)[0, 1]

        # Print epoch results
        print(
            f"Epoch [{epoch+1}/{train_cfg['epochs']}]  "
            f"Train Loss (recon/pocket/b_aff): {avg_tr_total_loss:.4f} ({avg_tr_recon_loss:.4f}/{avg_tr_pocket_loss:.4f}/{avg_tr_aff_loss:.4f})  "
            f"Valid Loss (recon/pocket/b_aff): {avg_vl_total_loss:.4f} ({avg_vl_recon_loss:.4f}/{avg_vl_pocket_loss:.4f}/{avg_vl_aff_loss:.4f})  "
            f"Valid vDCC_{train_cfg['DCC_threshold']}: {avg_vl_vDCC:.4f}  "
            f"Valid vDCC_{train_cfg['DCC_threshold']}_SR_{train_cfg['DCC_SR_threshold']}: {avg_vl_vDCC_SR:.4f}  "
            f"Valid DVO_{train_cfg['DVO_threshold']}: {avg_vl_DVO:.4f}  "
            f"Valid DCC_nan_count: {vl_DCC_nan_count:.4f}  "
            f"Valid PCC: {vl_PCC:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss/train/total": avg_tr_total_loss,
                "loss/train/reconstruction": avg_tr_recon_loss,
                "loss/train/pocket": avg_tr_pocket_loss,
                "loss/train/affinity": avg_tr_aff_loss,
                "loss/valid/total": avg_vl_total_loss,
                "loss/valid/reconstruction": avg_vl_recon_loss,
                "loss/valid/pocket": avg_vl_pocket_loss,
                "loss/valid/affinity": avg_vl_aff_loss,
                f"metrics/valid/vDCC_theta{train_cfg['DCC_threshold']}": avg_vl_vDCC,
                f"metrics/valid/vDCC_theta{train_cfg['DCC_threshold']}_SR_{train_cfg['DCC_SR_threshold']}": avg_vl_vDCC_SR,
                f"metrics/valid/DVO_theta{train_cfg['DVO_threshold']}": avg_vl_DVO,
                "metrics/valid/DCC_nan_count": vl_DCC_nan_count,
                "metrics/valid/PCC": vl_PCC,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # scheduler.step(avg_vl_total_loss)
        scheduler.step()

        # Save best model
        if avg_vl_total_loss < best_vl_loss:
            best_vl_loss = avg_vl_total_loss
            best_model = model.state_dict().copy()
            # best metric 기록
            best_epoch = epoch + 1
            best_vl_loss = avg_vl_total_loss
            best_vl_recon_loss = avg_vl_recon_loss
            best_vl_pocket_loss = avg_vl_pocket_loss
            best_vl_aff_loss = avg_vl_aff_loss
            best_vl_vDCC = avg_vl_vDCC
            best_vl_vDCC_SR = avg_vl_vDCC_SR
            best_vl_DVO = avg_vl_DVO
            best_DCC_nan_count = vl_DCC_nan_count
            best_vl_PCC = vl_PCC
            print("Best model has saved!")

            # 최상의 모델을 W&B에 저장
            # model_artifact = wandb.Artifact(
            #     name=f"best_model_run_{wandb.run.id}",
            #     type="model",
            #     description=f"Best model from epoch {best_epoch}",
            # )
            os.makedirs(train_cfg["save_dir"], exist_ok=True)
            torch.save(
                best_model,
                os.path.join(train_cfg["save_dir"], f"{exp_name}.pt"),
            )
            # model_artifact.add_file(
            #     os.path.join(train_cfg["output_dir"], f"{experiment_name}.pt")
            # )
            # wandb.log_artifact(model_artifact)

        # Early stopping check
        end_epoch = train_cfg["epochs"]
        improved = early_stopping(avg_vl_total_loss, epoch + 1)
        if early_stopping.early_stop:
            end_epoch = epoch + 1
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"Best model was at epoch {early_stopping.best_epoch}")
            break

    # Load best model
    model.load_state_dict(best_model)

    # writer 닫기
    if wandb.run is not None:
        wandb.run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_total_loss": best_vl_loss,
                "best_recon_loss": best_vl_recon_loss,
                "best_pocket_loss": best_vl_pocket_loss,
                "best_aff_loss": best_vl_aff_loss,
                "best_vDCC": best_vl_vDCC,
                "best_vDCC_SR": best_vl_vDCC_SR,
                "best_DVO": best_vl_DVO,
                "best_DCC_nan_count": best_DCC_nan_count,
                "best_PCC": best_vl_PCC,
                "avg_nan_count": sum(nan_count_ls) / len(nan_count_ls),
            }
        )

    wandb.finish()

    avg_nan_count = sum(nan_count_ls) / len(nan_count_ls)

    num_metrics = [
        best_vl_loss,
        best_vl_recon_loss,  # 숫자로 구성된 metrics들
        best_vl_pocket_loss,
        best_vl_aff_loss,
        best_vl_vDCC,
        best_vl_vDCC_SR,
        best_vl_DVO,
        best_DCC_nan_count,
        best_vl_PCC,
        avg_nan_count,
    ]

    return_metrics = [f"{best_epoch}/{end_epoch}"] + [
        round(num, 4) for num in num_metrics
    ]

    return model, return_metrics


def save_results(model: nn.Module, metrics: dict, exp_name: str, save_dir: str):
    # 여기서 성능 요약 출력하고, 저장하고, 어디에 저장했는지까지 로깅
    pass


def main():
    data_cfg, train_cfg = get_configs()

    if train_cfg["seed"] != -1:
        fix_seed(train_cfg["seed"])

    mdl = MasterDataLoader(data_cfg, train_cfg["seed"], train_cfg["batch_size"], train_cfg["num_workers"])
    tr_loader, vl_loader = mdl.get_tr_vl_loader()
    
    model = InSiteDTA(
        spatial_dims=3,
        in_channels=train_cfg["in_channels"],
        out_channels=train_cfg["out_channels"],
        feature_size=train_cfg["feature_size"],
        sw_depths=train_cfg["sw_depths"],
        sw_window_size=train_cfg["sw_window_size"],
        sw_patch_size=train_cfg["sw_patch_size"],
        sw_num_heads=train_cfg["sw_num_heads"],
        sw_mlp_ratio=train_cfg["sw_mlp_ratio"],
        sw_qkv_bias=train_cfg["sw_qkv_bias"],
        sw_drop_rate=train_cfg["sw_drop_rate"],
        sw_attn_drop_rate=train_cfg["sw_attn_drop_rate"],
        sw_drop_path_rate=train_cfg["sw_drop_path_rate"],
        sw_act=train_cfg["sw_act"],
        mol_encoder_types=train_cfg["mol_encoder_types"],
        mol_in_channels=train_cfg["mol_in_channels"],
        mol_hidden_channels=train_cfg["mol_hidden_channels"],
        mol_out_channels=train_cfg["mol_out_channels"],
        mol_num_layers=train_cfg["mol_num_layers"],
        mol_num_interactions_3d=train_cfg["mol_num_interactions_3d"],
        mol_dropout_3d=train_cfg["mol_dropout_3d"],
        mol_cutoff_3d=train_cfg["mol_cutoff_3d"],
        mol_num_filters_schnet=train_cfg["mol_num_filters_schnet"],
        mol_edge_num_gaussian_schnet=train_cfg["mol_edge_num_gaussian_schnet"],
        mol_edge_num_fourier_feats_egnn=train_cfg["mol_edge_num_fourier_feats_egnn"],
        mol_soft_edge_egnn=train_cfg["mol_soft_edge_egnn"],
        mol_act=train_cfg["mol_act"],
        dec_drop_rate=train_cfg["dec_drop_rate"],
        dec_act=train_cfg["dec_act"],
        ca_num_heads=train_cfg["ca_num_heads"],
        ca_dropout=train_cfg["ca_dropout"],
    )

    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    exp_name = (
        f"{timestamp}_{os.path.basename(train_cfg['data_config']).replace('.json', '')}"
    )

    model, metrics = train_model(
        model,
        tr_loader,
        vl_loader,
        train_cfg,
        voxel_size=data_cfg["voxel_size"],
        exp_name=exp_name,
    )

    save_results(model, metrics, exp_name=exp_name, save_dir=train_cfg["save_dir"])

    return exp_name


if __name__ == "__main__":
    main()
