# TODO: implement an ensemble inference with three checkpoint

import argparse
import numpy as np

import torch
from torch_geometric.data import Batch

from src.scripts.model.model import InSiteDTA
from src.scripts.preprocess.generate_mol_object import generate_mol_object, generate_conformers
from src.scripts.preprocess.ligand_featurization import encode_ligand_to_Data
from src.scripts.preprocess.protein_voxelization import ProteinVoxelizer


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to protein PDB file")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the ligand")
    parser.add_argument("--ckpt", type=str, default="./src/ckpt/run_2.pt", help="Path to model checkpoint file")
    return parser.parse_args()


def prep_single_smiles(smiles):
    m = generate_mol_object(smiles)
    if m is None:
        raise RuntimeError(f"Mol object was not created with smiles '{smiles}'")
    m = generate_conformers(m, target_numConfs=5)
    
    return m


def prep_single_protein(pdb_path: str, device="cuda:0"):
    pv = ProteinVoxelizer(voxel_size=2, n_voxels=32)
    voxel, center = pv.voxelize_inference(
        protein_path = pdb_path
    )
    return voxel, center


def inference_single(voxel, mol, ckpt, device):
    voxel = voxel.astype(np.float32)
    voxel = torch.from_numpy(voxel).unsqueeze(0).permute(0, 4, 1, 2, 3)
    voxel = voxel.to(device)
    
    lig_data = encode_ligand_to_Data(mol)
    lig_data = Batch.from_data_list([lig_data])
    lig_data = lig_data.to(device)
        
    model = InSiteDTA(out_channels=1)
    model.load_state_dict(torch.load(ckpt, weights_only=False))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        pred_poc, pred_aff = model(voxel, lig_data)
    
    return pred_aff, pred_poc


def main():
    args = get_arguments()
    voxel, center = prep_single_protein(args.pdb_path)
    mol = prep_single_smiles(smiles=args.smiles)
    pred_aff, pred_poc = inference_single(voxel, mol, ckpt=args.ckpt, device="cuda:0")
    print(f"Predicted Binding Affinity: {round(pred_aff.item(),4)} (pK)")
    
if __name__ == "__main__":
    main()