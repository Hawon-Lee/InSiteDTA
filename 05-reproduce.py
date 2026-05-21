import argparse, os, pickle, json
import numpy as np
import pandas as pd

from typing import Literal
from tqdm import tqdm

import torch
from torch_geometric.data import Batch

from src.scripts.model.model import InSiteDTA
from src.scripts.preprocess.generate_mol_object import generate_mol_object, generate_conformers
from src.scripts.preprocess.ligand_featurization import encode_ligand_to_Data
from src.scripts.preprocess.protein_voxelization import ProteinVoxelizer
from src.scripts.utils_inference import calc_metrics
from src.scripts.utils import print_args
from src.scripts.utils_train import calc_DCC_with_logit, calc_DVO_with_logit


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["crystal", "redocked", "p2rank", "alphafold"], required=True, help="Coreset scenario to evaluate")
    parser.add_argument("--ckpt", type=str, nargs="+", required=True, help="Path(s) to model checkpoint(s)")
    parser.add_argument("--batch_size", type=int, default=64, help="Bacth size for inference")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use")
    parser.add_argument("--center_method", type=str, default="protein", choices=["intelligent", "protein"], help="Voxel grid center: 'intelligent' (shift toward pocket) or 'protein' (protein geometric center)")
    parser.add_argument("--label_radius", type=float, default=0, help="Radius for pocket labeling in voxelization (0 for exact voxel matching)")
    parser.add_argument("--esm_dir", type=str, default=None, help="Directory containing pre-extracted ESM embedding pkl files")
    parser.add_argument("--pca_path", type=str, default=None, help="Path to PCA model pkl for ESM dim reduction")
    parser.add_argument("--use_tta", action="store_true", help="Enable 6-face TTA: average affinity predictions across 6 axis-aligned rotations of the voxel.")
    parser.add_argument("--dcc_sr_cutoff", type=float, default=4.0, help="DCC success-rate cutoff in Å (DCC_SR@<cutoff>Å).")
    parser.add_argument("--pocket_threshold", type=float, default=0.5, help="Sigmoid threshold for binarizing predicted pocket logits (used by DCC/DVO).")
    return parser.parse_args()


# 6 axis-aligned rotations on voxel [B, C, D, H, W]; affinity is invariant to spatial rotation.
# For pocket-prediction TTA we apply the inverse rotation per face so the predictions are aligned
# back to the original frame before averaging.
FACE_ROTATIONS = [
    lambda x: x,                          # 0: front (identity)
    lambda x: x.flip(4),                  # 1: back
    lambda x: x.transpose(2, 4),          # 2: left
    lambda x: x.transpose(2, 4).flip(4),  # 3: right
    lambda x: x.transpose(3, 4),          # 4: top
    lambda x: x.transpose(3, 4).flip(4),  # 5: bottom
]

FACE_INV_ROTATIONS = [
    lambda x: x,                          # 0: identity
    lambda x: x.flip(4),                  # 1: back-1 (self-inverse)
    lambda x: x.transpose(2, 4),          # 2: left-1 (self-inverse)
    lambda x: x.flip(4).transpose(2, 4),  # 3: right-1 (apply rev order)
    lambda x: x.transpose(3, 4),          # 4: top-1 (self-inverse)
    lambda x: x.flip(4).transpose(3, 4),  # 5: bottom-1
]


def prep_ligand(smi_csv, input_dir="./model_input"):
    os.makedirs(f"{input_dir}/ligands", exist_ok=True)
    smi_df = pd.read_csv(smi_csv)
    # ligand preparation
    for _, rows in tqdm(smi_df.iterrows(), total=len(smi_df), desc="1. Preparing ligands"):
        pdb_id = rows['PDB_ID']
        smi = rows['Canonical SMILES']
        out_path = f"{input_dir}/ligands/{pdb_id}_ligand.pkl"
        if os.path.exists(out_path):
            continue
        
        m = generate_mol_object(smi)
        if m is None: raise RuntimeError(f"Mol object was not created with smiles '{smi}'")
        m = generate_conformers(m, target_numConfs=5)
        
        with open(out_path, 'wb') as fp:
            pickle.dump(m, fp)

def prep_protein(data_dir, input_dir="./model_input", device="cuda:0", scenario="crystal", center_method="intelligent", label_radius=2.0, esm_dir=None, pca_path=None):
    os.makedirs(f"{input_dir}/proteins_{scenario}", exist_ok=True)
    pdb_id_ls = sorted(os.listdir(data_dir))
    pv = ProteinVoxelizer(voxel_size=2, n_voxels=32)

    # Load PCA model if provided
    pca_model = None
    if pca_path is not None:
        with open(pca_path, "rb") as fp:
            pca_model = pickle.load(fp)
        print(f"  Loaded PCA model ({pca_model['n_components']} dims) from {pca_path}")

    # protein preparation
    for pdb_id in tqdm(pdb_id_ls, desc="2. Preparing proteins"):
        ptn_path = f"{data_dir}/{pdb_id}/{pdb_id}_protein.pdb"
        poc_path = f"{data_dir}/{pdb_id}/{pdb_id}_pocket.pdb"
        out_data_name = os.path.join(f"{input_dir}/proteins_{scenario}/{pdb_id}_voxel.pkl")
        out_center_name = os.path.join(f"{input_dir}/proteins_{scenario}/{pdb_id}_center.pkl")

        if os.path.exists(out_data_name) and os.path.exists(out_center_name):
            continue

        # Load ESM embedding if available
        esm_data = None
        if esm_dir is not None:
            esm_path = os.path.join(esm_dir, f"{pdb_id}_esm.pkl")
            if os.path.exists(esm_path):
                with open(esm_path, "rb") as fp:
                    esm_data = pickle.load(fp)

        defined_center = None
        if center_method == "protein":
            defined_center = pv.calc_protein_center(ptn_path)

        voxel, label, center = pv.voxelize_gpu_v2(
                            protein_path=ptn_path,
                            pocket_path=poc_path,
                            r_cutoff=4.0,
                            device=device,
                            batch_size=8192,
                            defined_center=defined_center,
                            label_radius=label_radius,
                            esm_data=esm_data,
                            pca_model=pca_model,
                        )

        protein_data = np.concatenate((voxel, label), axis=3).astype(np.float16)
        with open(out_data_name, "wb") as fp:
            pickle.dump(protein_data, fp)

        with open(out_center_name, "wb") as fp:
            pickle.dump(center, fp)

def inference(lig_dir="./model_input/ligands", ptn_dir="./model_input/proteins",
              device="cuda:0", batch_size=64, index=None, ckpt=None, desc=None,
              use_tta=False, pocket_threshold=0.5, dcc_sr_cutoff=4.0, voxel_size=2):
    """Run inference for one checkpoint. Returns:
        pred_aff   : Tensor (N,) — affinity predictions
        target_aff : list[float] — true affinities
        dcc_mean   : mean DCC over samples with non-empty pocket prediction (Å)
        dcc_sr     : success rate (DCC < dcc_sr_cutoff Å); empty-prediction samples count as failures
        dvo_mean   : mean DVO over all samples
        nan_count  : number of samples whose predicted pocket was empty
    """
    _get_paths = lambda x: [os.path.join(x, f) for f in sorted(os.listdir(x)) if f.endswith("_ligand.pkl") or f.endswith("_voxel.pkl")]
    _crop_ids = lambda x: os.path.basename(x).split("_")[0]

    lig_paths = _get_paths(lig_dir)
    ptn_paths = _get_paths(ptn_dir)

    lig_map = {_crop_ids(l): l for l in lig_paths}
    ptn_map = {_crop_ids(p): p for p in ptn_paths}

    common_keys = sorted(set(lig_map.keys()) & set(ptn_map.keys()))
    n_lig_only = len(lig_map) - len(common_keys)
    n_ptn_only = len(ptn_map) - len(common_keys)

    if not common_keys:
        raise RuntimeError("No matching ligand/protein pairs found.")
    if n_lig_only > 0 or n_ptn_only > 0:
        print(f"  [INFO] lig={len(lig_map)}, ptn={len(ptn_map)} → {len(common_keys)} paired ({n_lig_only} lig-only, {n_ptn_only} ptn-only skipped)")

    lig_paths = [lig_map[k] for k in common_keys]
    ptn_paths = [ptn_map[k] for k in common_keys]

    # load index
    total_target_ba = []
    with open(index, "r") as fp:
        index = json.load(fp)
    for p in common_keys:
        total_target_ba.append(index[p])

    # ligand load & featurization
    lig_feat_ls = []
    for lig in lig_paths:
        with open(lig, "rb") as fp:
            m = pickle.load(fp)
            lig_feat_ls.append(encode_ligand_to_Data(m))

    lig_batch_ls = []
    for i in range(0, len(lig_feat_ls), batch_size):
        lig_batch = lig_feat_ls[i: i+batch_size]
        lig_batch = Batch.from_data_list(lig_batch).to(device)
        lig_batch_ls.append(lig_batch)

    # protein load — keep both input voxel (no label) and pocket label channel
    ptn_feat_ls = []
    for ptn in ptn_paths:
        with open(ptn, "rb") as fp:
            ptn_feat = pickle.load(fp).astype(np.float32)
            ptn_feat_ls.append(ptn_feat)

    ptn_batch_ls = []
    pocket_batch_ls = []
    for i in range(0, len(ptn_feat_ls), batch_size):
        ptn_stack = np.stack(ptn_feat_ls[i: i+batch_size]).astype(np.float32)
        ptn_stack = torch.from_numpy(ptn_stack).to(device)
        # last channel = pocket label (binary)
        pocket_batch = ptn_stack[..., -1:].permute(0, 4, 1, 2, 3)
        ptn_batch = ptn_stack[..., :-1].permute(0, 4, 1, 2, 3)
        ptn_batch_ls.append(ptn_batch)
        pocket_batch_ls.append(pocket_batch)

    model = InSiteDTA(out_channels=1)
    model.load_state_dict(torch.load(ckpt, weights_only=False))
    model.to(device)
    model.eval()
    pred_ba_ls = []
    dcc_ls = []
    dvo_ls = []
    nan_count = 0
    total_count = 0

    with torch.no_grad():
        for lig_batch, ptn_batch, pocket_batch in tqdm(
            zip(lig_batch_ls, ptn_batch_ls, pocket_batch_ls),
            total=len(lig_batch_ls), desc=desc
        ):
            if use_tta:
                aff_preds = []
                poc_preds = []
                for j, rot_fn in enumerate(FACE_ROTATIONS):
                    rot_ptn = rot_fn(ptn_batch)
                    pred_pocket_rot, pred_ba = model(rot_ptn, lig_batch)
                    if pred_ba.dim() == 0:
                        pred_ba = pred_ba.unsqueeze(0)
                    aff_preds.append(pred_ba)
                    # invert rotation so all pocket logits live in the original frame
                    poc_preds.append(FACE_INV_ROTATIONS[j](pred_pocket_rot))
                pred_ba     = torch.stack(aff_preds, dim=0).mean(dim=0)
                pred_pocket = torch.stack(poc_preds, dim=0).mean(dim=0)
            else:
                pred_pocket, pred_ba = model(ptn_batch, lig_batch)
                if pred_ba.dim() == 0:
                    pred_ba = pred_ba.unsqueeze(0)
            pred_ba_ls.append(pred_ba)

            # pocket metrics (DCC, DVO) per batch
            dccs, nan_idx = calc_DCC_with_logit(
                pred_pocket, pocket_batch,
                voxel_size=voxel_size, threshold=pocket_threshold,
            )
            dvos = calc_DVO_with_logit(
                pred_pocket, pocket_batch, threshold=pocket_threshold,
            )
            dcc_ls.append(dccs.detach().cpu())
            dvo_ls.append(dvos.detach().cpu())
            nan_count   += len(nan_idx)
            total_count += pocket_batch.shape[0]

    total_pred_ba = torch.concat(pred_ba_ls).cpu()
    total_dccs    = torch.cat(dcc_ls) if dcc_ls else torch.zeros(0)
    total_dvos    = torch.cat(dvo_ls) if dvo_ls else torch.zeros(0)

    dcc_mean = float(total_dccs.float().mean()) if total_dccs.numel() else float("nan")
    # SR denominator includes nan (failure-to-predict) samples
    dcc_sr   = float((total_dccs < dcc_sr_cutoff).sum()) / max(total_count, 1)
    dvo_mean = float(total_dvos.float().mean()) if total_dvos.numel() else float("nan")

    return total_pred_ba, total_target_ba, dcc_mean, dcc_sr, dvo_mean, nan_count

def main():
    args = get_arguments()
    scenario = args.scenario
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"; args.device = device
    batch_size = args.batch_size
    print_args(args)
    
    index = "./src/data/index/affinity_index_pdbbind2020.json"
    ckpt_ls = args.ckpt

    smi_csv = "./src/data/index/ligand_smiles_coreset.csv"
    data_dir = f"./src/data/coreset_{scenario}"
    input_dir = f"./model_input"

    prep_ligand(smi_csv=smi_csv, input_dir=input_dir)
    prep_protein(data_dir=data_dir, input_dir=input_dir, device=device, scenario=scenario, center_method=args.center_method, label_radius=args.label_radius, esm_dir=args.esm_dir, pca_path=args.pca_path)

    aggr_results = {'pcc': [], 'rmse': [], 'mae': [],
                    'dcc': [], 'dcc_sr': [], 'dvo': []}
    nan_per_ckpt = []
    for i, ckpt in enumerate(ckpt_ls):
        tta_tag = " [TTA-6faces]" if args.use_tta else ""
        desc = f"3-{i+1}. Evaluating InSiteDTA{tta_tag} ({i+1}/{len(ckpt_ls)}) on coreset_{scenario}"
        pred, target, dcc, dcc_sr, dvo, n_nan = inference(
            lig_dir=f"{input_dir}/ligands",
            ptn_dir=f"{input_dir}/proteins_{scenario}",
            batch_size=batch_size, device=device, index=index, ckpt=ckpt, desc=desc,
            use_tta=args.use_tta,
            pocket_threshold=args.pocket_threshold,
            dcc_sr_cutoff=args.dcc_sr_cutoff,
        )
        pcc, rmse, mae = calc_metrics(pred, target)
        # per-ckpt pocket metrics (calc_metrics already printed PCC/RMSE/MAE)
        print(f"  DCC : {round(dcc, 4)} Å")
        print(f"  SR@{args.dcc_sr_cutoff:g}Å: {round(dcc_sr, 4)}")
        print(f"  DVO : {round(dvo, 4)}")
        if n_nan > 0:
            print(f"  (empty pocket prediction: {n_nan})")
        aggr_results['pcc'].append(pcc)
        aggr_results['rmse'].append(rmse)
        aggr_results['mae'].append(mae)
        aggr_results['dcc'].append(dcc)
        aggr_results['dcc_sr'].append(dcc_sr)
        aggr_results['dvo'].append(dvo)
        nan_per_ckpt.append(n_nan)

    metric_labels = {
        'pcc':   'PCC',
        'rmse':  'RMSE',
        'mae':   'MAE',
        'dcc':   'DCC (Å)',
        'dcc_sr': f'DCC_SR@{args.dcc_sr_cutoff:g}Å',
        'dvo':   'DVO',
    }
    tta_summary = " (with 6-face TTA)" if args.use_tta else ""
    print(f"4. Aggregated results on {len(ckpt_ls)} different random seeds{tta_summary}:")
    for metric, score_ls in aggr_results.items():
        mean = np.array(score_ls).mean()
        std = np.array(score_ls).std(ddof=1) if len(score_ls) > 1 else 0.0
        print(f"- {metric_labels[metric]}: {round(mean, 3)} ± {round(std, 3)}")
    if any(n > 0 for n in nan_per_ckpt):
        print(f"  (empty pocket prediction count per ckpt: {nan_per_ckpt})")
    
if __name__ == "__main__":
    main()