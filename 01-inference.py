import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from src.scripts.model.model import InSiteDTA
from src.scripts.preprocess.generate_mol_object import generate_conformers, generate_mol_object
from src.scripts.preprocess.ligand_featurization import encode_ligand_to_Data
from src.scripts.preprocess.protein_voxelization import ProteinVoxelizer
from src.scripts.utils import print_args
from src.scripts.utils_inference import P2RankRunner


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to protein PDB file")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the ligand")
    parser.add_argument("--ckpt", type=str, default="./src/ckpt/run_2.pt", help="Path to model checkpoint file")
    parser.add_argument("--use_p2rank", action="store_true", help="Use P2Rank to guide protein voxelization")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--save_bs_pdb", type=str, default=None,
                        help="If set, save residue-level binding site PDB to this path")
    parser.add_argument("--save_voxel_pdb", type=str, default=None,
                        help="If set, save voxel-center dummy-atom PDB (visual grid) to this path")
    parser.add_argument("--bs_threshold", type=float, default=0.5,
                        help="Probability threshold for binding site extraction (default 0.5)")
    return parser.parse_args()


def prep_single_smiles(smiles):
    m = generate_mol_object(smiles)
    if m is None:
        raise RuntimeError(f"Mol object was not created with smiles '{smiles}'")
    m = generate_conformers(m, target_numConfs=5)
    return m


def prep_single_protein(pdb_path: str, use_p2rank: bool = False, device="cuda:0"):
    pv = ProteinVoxelizer(voxel_size=2, n_voxels=32)
    if use_p2rank:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = P2RankRunner()
            csv_path = runner.run_p2rank(pdb_path, tmpdir)
            pocket_path = runner.p2rank_res_to_pdb(csv_path, pdb_path, tmpdir)
            voxel, _, center = pv.voxelize_gpu_v2(
                protein_path=pdb_path, pocket_path=pocket_path, device=device,
            )
    else:
        voxel, center = pv.voxelize_inference(protein_path=pdb_path, device=device)
    return voxel, center, pv


def inference_single(voxel, mol, ckpt, device):
    voxel = voxel.astype(np.float32)
    voxel = torch.from_numpy(voxel).unsqueeze(0).permute(0, 4, 1, 2, 3).to(device)

    lig_data = encode_ligand_to_Data(mol)
    lig_data = Batch.from_data_list([lig_data]).to(device)

    model = InSiteDTA(out_channels=1)
    model.load_state_dict(torch.load(ckpt, weights_only=False))
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_logits, pred_aff = model(voxel, lig_data)
    pred_pocket_prob = torch.sigmoid(pred_logits[:, -1:, ...])
    return pred_aff, pred_pocket_prob


def save_bs_residue_pdb(voxelizer: ProteinVoxelizer, protein_path: str, pred_prob: torch.Tensor,
                        center: np.ndarray, threshold: float, save_path: str) -> int:
    """Save residue-level binding site PDB. Returns number of residues."""
    atoms = voxelizer.get_predicted_pocket_atoms_from_pred(
        protein_path, pred_prob, center=center, threshold=threshold
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if not atoms:
        Path(save_path).write_text("")
        return 0
    pdb_content = voxelizer.get_pocket_aminoacids(protein_path, atoms)
    Path(save_path).write_text(pdb_content)
    return len({a[3] for a in atoms})


def save_bs_voxel_pdb(voxelizer: ProteinVoxelizer, pred_prob: torch.Tensor,
                      center: np.ndarray, threshold: float, save_path: str) -> int:
    """Save voxel-center dummy-atom PDB visualizing the raw prob>threshold grid."""
    if isinstance(pred_prob, torch.Tensor):
        arr = pred_prob.detach().cpu().numpy()
    else:
        arr = pred_prob
    while arr.ndim > 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[-1] if arr.shape[0] < arr.shape[-1] else arr[..., -1]

    grid_size = voxelizer.voxel_size * voxelizer.n_voxels
    start = np.asarray(center) - grid_size / 2
    voxel_centers = voxelizer.get_voxel_centers(start)

    mask = arr > threshold
    coords = voxel_centers[mask]
    probs = arr[mask]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (xyz, p) in enumerate(zip(coords, probs), start=1):
        x, y, z = xyz
        b = float(p) * 100.0
        lines.append(
            f"HETATM{i:>5d}  O   BS  A{i:>4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           O\n"
        )
    lines.append("END\n")
    Path(save_path).write_text("".join(lines))
    return int(mask.sum())


def main():
    args = get_arguments()
    print_args(args)
    device = f"cuda:{args.device}"

    voxel, center, voxelizer = prep_single_protein(args.pdb_path, use_p2rank=args.use_p2rank, device=device)
    mol = prep_single_smiles(smiles=args.smiles)
    pred_aff, pred_prob = inference_single(voxel, mol, ckpt=args.ckpt, device=device)

    print(f"Predicted Binding Affinity: {round(pred_aff.item(), 4)} (pK)")

    if args.save_bs_pdb:
        n_res = save_bs_residue_pdb(
            voxelizer, args.pdb_path, pred_prob, center, args.bs_threshold, args.save_bs_pdb,
        )
        print(f"Saved binding site residues ({n_res} residues) -> {args.save_bs_pdb}")

    if args.save_voxel_pdb:
        n_vox = save_bs_voxel_pdb(
            voxelizer, pred_prob, center, args.bs_threshold, args.save_voxel_pdb,
        )
        print(f"Saved binding site voxel dummy atoms ({n_vox} voxels) -> {args.save_voxel_pdb}")


if __name__ == "__main__":
    main()
