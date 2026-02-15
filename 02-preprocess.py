# TODO: flatten data structure 구현 (단백질 폴더에 .pdb 만 있고(pocket 유무도 고려) / 리간드는 무조건 ligand smiles 형태로 제시)
# TODO: implement first-channel representation

import os, sys, argparse, pickle
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from typing import Literal


from src.scripts.preprocess.protein_voxelization import ProteinVoxelizer
from src.scripts.preprocess.generate_mol_object import generate_mol_object, generate_conformers

def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--raw_dir", type=str, default="src/data/samples", help="Directory containing raw protein PDB files")
    parser.add_argument("--save_dir", type=str, default="model_input", help="Output directory for preprocessed protein and ligand files")

    # ligand featurization args
    parser.add_argument("--smiles_csv", type=str, default="src/data/index/ligand_smiles_pdbbind2020.csv", help="Csv file path containing smiles with pdb id")
    
    # protein voxelization args
    parser.add_argument(
        "--data_structure",
        type=str,
        default="nested",
        choices=["nested", "flatten"],
        help="Structure of the raw pdb data directory (pdbbind format: nested)"
    )
    
    
    # voxelization configs
    parser.add_argument("--voxel_size", type=int, default=2, help="Size of each voxel in Angstroms")
    parser.add_argument("--n_voxels", type=int, default=32, help="Number of voxels along each dimension of the 3D grid")
    parser.add_argument("--device", default=0, help="GPU index to use while protein voxelization")

    # # Dataset selection
    # parser.add_argument(
    #     "--train_source_name",
    #     type=str,
    #     default="PDBbind2020",
    #     choices=[
    #         "PDBbind2020",
    #         "scPDB",
    #         "Davis",
    #         "Kiba",
    #         "Filtered_Davis",
    #         "PDBbind2016",
    #         "Da_Ki",
    #         "PDBbind_Da_Ki",
    #     ],
    #     help="Target source dataset for training and validation",
    # )
    # parser.add_argument(
    #     "--test_source_name",
    #     type=str,
    #     default="PDBbind2020",
    #     choices=[
    #         "PDBbind2020",
    #         "scPDB",
    #         "Davis",
    #         "Kiba",
    #         "Filtered_Davis",
    #         "PDBbind2016",
    #         "Da_Ki",
    #         "PDBbind_Da_Ki",
    #     ],
    #     help="Target dataset for evaluation model performance",
    # )
    # parser.add_argument(
    #     "--test_key_file_path",
    #     type=str,
    #     default="/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/DATA/data_split/test_keys/PDBbind-coreset.txt",
    #     help="Path to file containing test set keys",
    # )
    # parser.add_argument(
    #     "--val_size",
    #     type=float,
    #     default=0.15,
    #     help="Fraction of training data to use for validation",
    # )
    # parser.add_argument(
    #     "--lig_file_format",
    #     type=str,
    #     default="sdf",
    #     choices=["sdf", "mol2", "rdmol"],
    #     help="Ligand file format",
    # )

    # # Training hyperparameters
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=312,
    #     help="Seed for reproduce data split. 0 for random",
    # )
    # parser.add_argument(
    #     "--voxel_size", type=int, default=2, help="Size of each voxel in Angstroms"
    # )
    # parser.add_argument(
    #     "--input_dim",
    #     type=int,
    #     default=32,
    #     help="Dimension of the input voxel grid (number of voxels per side)",
    # )

    # # Dataset directories
    # parser.add_argument(
    #     "--smi_emb_dir",
    #     type=str,
    #     default="/data/hawon/Interaction_free/ligands_chemberta2",
    #     help="Directory containing preprocessed smiles embedding file (*.pt)",
    # )

    # parser.add_argument(
    #     "--PDBbind2020_lig_dir",
    #     type=str,
    #     default="/data/hawon/Interaction_free/ligands_from_can",
    #     help="Directory containing ligand structure files of PDBbind v2020",
    # )
    # parser.add_argument(
    #     "--PDBbind2020_pdb_dir",
    #     type=str,
    #     default="/data/hawon/Datasets/PDBbind2020_v1/General_plus_refined",
    #     help="Directory containing protein structure files of PDBbind v2020",
    # )
    # parser.add_argument(
    #     "--PDBbind2020_vox_dir",
    #     type=str,
    #     default="/data/hawon/Interaction_free/voxelized_pdbbind/voxel_size2_n_voxels32_new",
    #     help="Directory containing voxelized PDBbind v2020 data",
    # )

    # # parser.add_argument("--PDBbind2016_pdb_dir", type=str, default="/data/hawon/Datasets/PDBbind2016_v1/General_plus_refined", help="Directory containing protein and ligand structure files of PDBbind v2016")
    # # parser.add_argument("--PDBbind2016_vox_dir", type=str, default="/data/hawon/Interaction_free/voxelized_pdbbind/voxel_size2_n_voxels32_new", help="Directory containing voxelized PDBbind v2016 data")

    # parser.add_argument(
    #     "--scPDB_lig_dir",
    #     type=str,
    #     default="/data/hawon/Datasets/scPDB/scPDB_aspdbformat",
    #     help="Directory containing ligand structure files of scPDB",
    # )
    # parser.add_argument(
    #     "--scPDB_vox_dir",
    #     type=str,
    #     default="/data/hawon/Interaction_free/voxelized_scPDB/voxel_size2_n_voxels32_new",
    #     help="Directory containing voxelized scPDB data",
    # )

    # parser.add_argument(
    #     "--Davis_lig_dir",
    #     type=str,
    #     default="/data/hawon/Datasets/Davis_Kiba/Dav_aspdbformat",
    #     help="Directory containing ligand structure files of Davis dataset",
    # )
    # parser.add_argument(
    #     "--Davis_vox_dir",
    #     type=str,
    #     default="/data/hawon/Interaction_free/voxelized_davis/voxel_size2_n_voxels32_new",
    #     help="Directory containing voxelized Davis dataset",
    # )

    # parser.add_argument(
    #     "--Kiba_lig_dir",
    #     type=str,
    #     default="/data/hawon/Datasets/Davis_Kiba/Kib_aspdbformat",
    #     help="Directory containing ligand structure files of Kiba dataset",
    # )
    # parser.add_argument(
    #     "--Kiba_vox_dir",
    #     type=str,
    #     default="/data/hawon/Interaction_free/voxelized_Kiba/voxel_size2_n_voxels32_new",
    #     help="Directory containing voxelized Kiba dataset",
    # )

    # parser.add_argument(
    #     "--index_dir",
    #     type=str,
    #     default="/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/DATA/index",
    #     help="Directory containing binding affinity index files of datasets",
    # )
    # parser.add_argument(
    #     "--save_dir",
    #     type=str,
    #     default="/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/DATA/data_split",
    #     help="Directory to save data split configuration files",
    # )

    # # Output and processing options
    # parser.add_argument(
    #     "--filter_sw_similarity",
    #     action="store_true",
    #     default=False,
    #     help="Filter test data based on protein Smith-Waterman similarity to training set",
    # )
    # parser.add_argument(
    #     "--sw_lib_dir",
    #     type=str,
    #     default="../smith-waterman",
    #     help="Path to Smith-Waterman library containing pyssw.py and libssw.so",
    # )
    # parser.add_argument(
    #     "--sw_cache_dir",
    #     type=str,
    #     default="../../cache",
    #     help="Directory path to the Smith-Waterman score cache file containing pre-calculated score",
    # )
    # parser.add_argument(
    #     "--sim_threshold",
    #     type=float,
    #     default=0.7,
    #     help="Smith-Waterman Similarity threshold for filtering train dataset (lower is stricter)",
    # )
    # parser.add_argument(
    #     "--keep_filtered",
    #     action="store_true",
    #     default=False,
    #     help="Include similarity-filtered items in test dataset",
    # )

    args = parser.parse_args()
    return args

    
def collect_pdb_ids(data_structure: Literal["nested", "flatten"], raw_dir: str) -> list[str]:
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    
    if data_structure == "nested":
        pdb_id_ls = sorted([f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))])
    elif data_structure == "flatten":
        pdb_id_ls = sorted(["_".join(f.split("_")[:-1]) for f in os.listdir(raw_dir) if f.endswith("protein.pdb")])
    
    print(f"0. Collected # {len(pdb_id_ls)} target pdb ids from raw_dir: {raw_dir}")
    return pdb_id_ls


def featurize_ligand(smiles_csv: str, pdb_id_ls: list[str], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    smiles_df = pd.read_csv(smiles_csv)

    for pdb_id in tqdm(pdb_id_ls, desc="1. Preparing ligands"):
        match_row = smiles_df[smiles_df['PDB_ID'] == pdb_id]
        if match_row.empty:
            print(f"[Warning] Skipping PDB ID '{pdb_id}' - not found in SMILES CSV")
            continue
        smi = match_row['Canonical SMILES'].iloc[0]
        out_name = f"{save_dir}/{pdb_id}_ligand.pkl"
        if os.path.exists(out_name):
            continue

        m = generate_mol_object(smi)
        if m is None: raise RuntimeError(f"Mol object was not created with smiles '{smi}'")
        m = generate_conformers(m, target_numConfs=5)
        
        with open(out_name, 'wb') as fp:
            pickle.dump(m, fp)


def voxelize_protein(data_structure: Literal["nested", "flatten"], raw_dir: str, pdb_id_ls: list[str], save_dir: str, voxel_size: int = 2, n_voxels: int = 32, device = 0) -> None:
    os.makedirs(save_dir, exist_ok=True)
    
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    
    # Path specification
    if data_structure == "nested":
        ptn_path_template = "{raw_dir}/{pdb_id}/{pdb_id}_protein.pdb"
        poc_path_template = "{raw_dir}/{pdb_id}/{pdb_id}_pocket.pdb"
        
    elif data_structure == "flatten":
        ptn_path_template = "{raw_dir}/{pdb_id}_protein.pdb"
        poc_path_template = "{raw_dir}/{pdb_id}_pocket.pdb"
        raise NotImplementedError("flatten mode is not supported yet")

    pv = ProteinVoxelizer(voxel_size=voxel_size, n_voxels=n_voxels)
    
    for pdb_id in tqdm(pdb_id_ls, desc="2. Preparing proteins"):
        ptn_path = ptn_path_template.format(raw_dir=raw_dir, pdb_id=pdb_id)
        poc_path = poc_path_template.format(raw_dir=raw_dir, pdb_id=pdb_id)
        out_voxel_name = f"{save_dir}/{pdb_id}_voxel.pkl"
        out_center_name = f"{save_dir}/{pdb_id}_center.pkl"
        
        if os.path.exists(out_voxel_name) and os.path.exists(out_center_name):
            continue
        
        voxel, label, center = pv.voxelize_gpu_v2(
                            protein_path=ptn_path,
                            pocket_path=poc_path,
                            r_cutoff=4.0,
                            device=device,
                            batch_size=8192
                        )
        
        # TODO: channel 위치 미리 조절
        protein_data = np.concatenate((voxel, label), axis=3).astype(np.float16)
        with open(out_voxel_name, "wb") as fp:
            pickle.dump(protein_data, fp)

        with open(out_center_name, "wb") as fp:
            pickle.dump(center, fp)


def generate_data_cfg() -> None:
    # match preprocessed protein and ligand files
    # print how many files were not successfully preprocessed
    pass


def main():
    args = get_arguments()
    
    pdb_id_ls = collect_pdb_ids(args.data_structure, args.raw_dir)
    
    featurize_ligand(
        smiles_csv=args.smiles_csv,
        pdb_id_ls=pdb_id_ls,
        save_dir=args.save_dir + "/input_ligand",
    )
    voxelize_protein(
        data_structure=args.data_structure,
        raw_dir=args.raw_dir,
        pdb_id_ls=pdb_id_ls,
        save_dir=args.save_dir + "/input_protein",
        voxel_size=args.voxel_size,
        n_voxels=args.n_voxels
    )
    # generate_data_cfg(args., args. ..., save_dir_ptn, save_dir_lig)
    
    pass

if __name__ == "__main__":
    main()