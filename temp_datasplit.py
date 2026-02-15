import os, sys
import numpy as np
import argparse
import subprocess
import tempfile
import json
import pickle
import multiprocessing as mp
from tqdm import tqdm
from glob import glob
from Bio.PDB.Polypeptide import three_to_one
from sklearn.model_selection import train_test_split
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Configure input data settings and dataset splits for model training and testing"
    )

    # Dataset selection
    parser.add_argument(
        "--train_source_name",
        type=str,
        default="PDBbind2020",
        choices=[
            "PDBbind2020",
            "scPDB",
            "Davis",
            "Kiba",
            "Filtered_Davis",
            "PDBbind2016",
            "Da_Ki",
            "PDBbind_Da_Ki",
        ],
        help="Target source dataset for training and validation",
    )
    parser.add_argument(
        "--test_source_name",
        type=str,
        default="PDBbind2020",
        choices=[
            "PDBbind2020",
            "scPDB",
            "Davis",
            "Kiba",
            "Filtered_Davis",
            "PDBbind2016",
            "Da_Ki",
            "PDBbind_Da_Ki",
        ],
        help="Target dataset for evaluation model performance",
    )
    parser.add_argument(
        "--test_key_file_path",
        type=str,
        default="/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/DATA/data_split/test_keys/PDBbind-coreset.txt",
        help="Path to file containing test set keys",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of training data to use for validation",
    )
    parser.add_argument(
        "--lig_file_format",
        type=str,
        default="sdf",
        choices=["sdf", "mol2", "rdmol"],
        help="Ligand file format",
    )

    # Training hyperparameters
    parser.add_argument(
        "--seed",
        type=int,
        default=312,
        help="Seed for reproduce data split. 0 for random",
    )
    parser.add_argument(
        "--voxel_size", type=int, default=2, help="Size of each voxel in Angstroms"
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=32,
        help="Dimension of the input voxel grid (number of voxels per side)",
    )

    # Dataset directories
    parser.add_argument(
        "--smi_emb_dir",
        type=str,
        default="/data/hawon/Interaction_free/ligands_chemberta2",
        help="Directory containing preprocessed smiles embedding file (*.pt)",
    )

    parser.add_argument(
        "--PDBbind2020_lig_dir",
        type=str,
        default="/data/hawon/Interaction_free/ligands_from_can",
        help="Directory containing ligand structure files of PDBbind v2020",
    )
    parser.add_argument(
        "--PDBbind2020_pdb_dir",
        type=str,
        default="/data/hawon/Datasets/PDBbind2020_v1/General_plus_refined",
        help="Directory containing protein structure files of PDBbind v2020",
    )
    parser.add_argument(
        "--PDBbind2020_vox_dir",
        type=str,
        default="/data/hawon/Interaction_free/voxelized_pdbbind/voxel_size2_n_voxels32_new",
        help="Directory containing voxelized PDBbind v2020 data",
    )

    # parser.add_argument("--PDBbind2016_pdb_dir", type=str, default="/data/hawon/Datasets/PDBbind2016_v1/General_plus_refined", help="Directory containing protein and ligand structure files of PDBbind v2016")
    # parser.add_argument("--PDBbind2016_vox_dir", type=str, default="/data/hawon/Interaction_free/voxelized_pdbbind/voxel_size2_n_voxels32_new", help="Directory containing voxelized PDBbind v2016 data")

    parser.add_argument(
        "--scPDB_lig_dir",
        type=str,
        default="/data/hawon/Datasets/scPDB/scPDB_aspdbformat",
        help="Directory containing ligand structure files of scPDB",
    )
    parser.add_argument(
        "--scPDB_vox_dir",
        type=str,
        default="/data/hawon/Interaction_free/voxelized_scPDB/voxel_size2_n_voxels32_new",
        help="Directory containing voxelized scPDB data",
    )

    parser.add_argument(
        "--Davis_lig_dir",
        type=str,
        default="/data/hawon/Datasets/Davis_Kiba/Dav_aspdbformat",
        help="Directory containing ligand structure files of Davis dataset",
    )
    parser.add_argument(
        "--Davis_vox_dir",
        type=str,
        default="/data/hawon/Interaction_free/voxelized_davis/voxel_size2_n_voxels32_new",
        help="Directory containing voxelized Davis dataset",
    )

    parser.add_argument(
        "--Kiba_lig_dir",
        type=str,
        default="/data/hawon/Datasets/Davis_Kiba/Kib_aspdbformat",
        help="Directory containing ligand structure files of Kiba dataset",
    )
    parser.add_argument(
        "--Kiba_vox_dir",
        type=str,
        default="/data/hawon/Interaction_free/voxelized_Kiba/voxel_size2_n_voxels32_new",
        help="Directory containing voxelized Kiba dataset",
    )

    parser.add_argument(
        "--index_dir",
        type=str,
        default="/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/DATA/index",
        help="Directory containing binding affinity index files of datasets",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/tech/Hawon/Develop/Interaction_free/MAIN_REFAC/new/DATA/data_split",
        help="Directory to save data split configuration files",
    )

    # Output and processing options
    parser.add_argument(
        "--filter_sw_similarity",
        action="store_true",
        default=False,
        help="Filter test data based on protein Smith-Waterman similarity to training set",
    )
    parser.add_argument(
        "--sw_lib_dir",
        type=str,
        default="../smith-waterman",
        help="Path to Smith-Waterman library containing pyssw.py and libssw.so",
    )
    parser.add_argument(
        "--sw_cache_dir",
        type=str,
        default="../../cache",
        help="Directory path to the Smith-Waterman score cache file containing pre-calculated score",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.7,
        help="Smith-Waterman Similarity threshold for filtering train dataset (lower is stricter)",
    )
    parser.add_argument(
        "--keep_filtered",
        action="store_true",
        default=False,
        help="Include similarity-filtered items in test dataset",
    )

    args = parser.parse_args()
    return args


def get_file_name(args):
    """Generate a configuration file name based on the input arguments."""
    test_data_name = os.path.basename(args.test_key_file_path).replace(".txt", "")

    config_name = f"datasplit_tr{args.train_source_name}_ts{test_data_name}_vox{args.voxel_size}_dim{args.input_dim}_lig{args.lig_file_format.upper()}"

    if args.filter_sw_similarity:
        config_name += f"_fil{args.sim_threshold}"
        if args.keep_filtered:
            config_name += "_incFiltered"

    # if args.seed == 0:
    #     config_name += f"_seedRandom"
    # else:
    #     config_name += f"_seed{args.seed}"

    return config_name + ".json"


def get_train_test_dir(args, return_index=True):

    index_path = None

    # get total data keys
    if args.train_source_name == "PDBbind2020":
        train_pdb_dir = args.PDBbind2020_pdb_dir
        train_vox_dir = args.PDBbind2020_vox_dir
        train_lig_dir = args.PDBbind2020_lig_dir
    elif args.train_source_name == "scPDB":
        train_pdb_dir = args.scPDB_pdb_dir
        train_vox_dir = args.scPDB_vox_dir
        train_lig_dir = args.scPDB_lig_dir
    elif args.train_source_name == "Davis":
        train_pdb_dir = args.Davis_pdb_dir
        train_vox_dir = args.Davis_vox_dir
        train_lig_dir = args.Davis_lig_dir
    elif args.train_source_name == "Filtered_Davis":
        train_pdb_dir = args.Filtered_Davis_pdb_dir
        train_vox_dir = args.Davis_vox_dir
        train_lig_dir = args.Davis_lig_dir
    elif args.train_source_name == "Kiba":
        train_pdb_dir = args.Kiba_pdb_dir
        train_vox_dir = args.Kiba_vox_dir
        train_lig_dir = args.Kiba_lig_dir
    else:
        print("Other train data formats are not supported yet.")
        return []

    # get total data keys
    if args.test_source_name == "PDBbind2020":
        test_pdb_dir = args.PDBbind2020_pdb_dir
        test_vox_dir = args.PDBbind2020_vox_dir
        test_lig_dir = args.PDBbind2020_lig_dir
    elif args.test_source_name == "scPDB":
        test_pdb_dir = args.scPDB_pdb_dir
        test_vox_dir = args.scPDB_vox_dir
        test_lig_dir = args.scPDB_lig_dir
    elif args.test_source_name == "Davis":
        test_pdb_dir = args.Davis_pdb_dir
        test_vox_dir = args.Davis_vox_dir
        test_lig_dir = args.Davis_lig_dir
    elif args.test_source_name == "Filtered_Davis":
        test_pdb_dir = args.Filtered_Davis_pdb_dir
        test_vox_dir = args.Davis_vox_dir
        test_lig_dir = args.Davis_lig_dir
    elif args.test_source_name == "Kiba":
        test_pdb_dir = args.Kiba_pdb_dir
        test_vox_dir = args.Kiba_vox_dir
        test_lig_dir = args.Kiba_lig_dir
    else:
        print("Other test data formats are not supported yet.")
        return []

    return (
        train_pdb_dir,
        train_vox_dir,
        test_pdb_dir,
        test_vox_dir,
        train_lig_dir,
        test_lig_dir,
    )


def read_test_keys(test_key_file_path):
    """Read test keys from the test key file."""
    if not os.path.exists(test_key_file_path):
        print(f"Warning: Test key file {test_key_file_path} not exists")
        return []

    with open(test_key_file_path, "r") as fp:
        test_keys = [line.strip() for line in fp.readlines() if line.strip()]

    print(f"Loaded {len(test_keys)} test keys from {test_key_file_path}")
    return test_keys


def get_pdb_path(pdb_key: str, pdb_dir: str):
    """Get path to the PDB file for a given key."""
    pdb_paths = glob(os.path.join(pdb_dir, pdb_key, "*protein.pdb"))

    if len(pdb_paths) > 1:
        print(f"Warning: {pdb_key} has more than two protein pdb files")
    elif len(pdb_paths) == 0:
        raise FileNotFoundError(f"Error: No protein PDB file found for {pdb_key}")

    return pdb_paths[0]


def get_pdb_sequence(
    pdb_path, standardize_residue=True, return_one_letter=True
) -> dict[str, str]:
    MOD2STD = {
        "MSE": "MET",  # 셀레노메티오닌 -> 메티오닌
        "SEP": "SER",  # 인산화세린 -> 세린
        "TPO": "THR",  # 인산화트레오닌 -> 트레오닌
        "PTR": "TYR",  # 인산화티로신 -> 티로신
        "HYP": "PRO",  # 하이드록시프롤린 -> 프롤린
        "KCX": "LYS",  # 리신-NZ-카복실산 -> 리신
        "CSD": "CYS",  # 시스테인설피닉산 -> 시스테인
        "MLY": "LYS",  # 메틸화리신 -> 리신
        "CSO": "CYS",  # 시스테인설피닉산 -> 시스테인
        "CSX": "CYS",  # 시스테인설폰산 -> 시스테인
        "CME": "CYS",  # S-카복시메틸시스테인 -> 시스테인
        "FME": "MET",  # N-포밀메티오닌 -> 메티오닌
        "LLP": "LYS",  # 리신-피리독살-5'-인산 -> 리신
        "PCA": "GLU",  # 피로글루탐산 -> 글루탐산
        "PYL": "LYS",  # 피롤리신 -> 리신
        "SEC": "CYS",  # 셀레노시스테인 -> 시스테인
        "STY": "TYR",  # 설포티로신 -> 티로신
    }

    STD = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]

    with open(pdb_path, "r") as fp:
        pdb_lines = fp.read().splitlines()

    chain_sequences = {}

    # 기본 chain 별 residue sequence parsing
    for line in pdb_lines:
        if line.startswith("SEQRES"):
            chain_id = line[11]
            residues = line.split()[4:]

            for residue in residues:
                if residue in list(MOD2STD.keys()) + STD:
                    if chain_id not in chain_sequences:
                        chain_sequences[chain_id] = []
                    chain_sequences[chain_id].append(residue)

    # option 적용
    for chain_id, sequence in chain_sequences.items():
        if standardize_residue:
            # 비표준 -> 표준 아미노산으로 변경
            sequence = [MOD2STD[aa] if aa in MOD2STD else aa for aa in sequence]

        if return_one_letter:
            try:
                sequence = "".join([three_to_one(aa) for aa in sequence])

            except KeyError as e:
                print(
                    f"Non-standard amino acid three-letter code {e} failed to convert to one-letter code. Use standardize_residue option."
                )
                return {}
        chain_sequences[chain_id] = sequence

    return chain_sequences


def calc_smith_waterman_score(
    seq1,
    seq2,
    src_path="../smith-waterman/",
    match_point=2,
    mismatch_penalty=2,
    gap_open_penalty=3,
    gap_ext_penalty=1,
    protein=True,
):
    """
    Calculate Smith-Waterman similarity score between two sequences (protein or nucleotide).

    This function uses the Complete Striped Smith-Waterman Library to compute sequence alignment
    similarity scores between two biological sequences.

    Args:
        src_path (str): Path to the Smith-Waterman calculator source library compiled for python interface (sholud inclued pyssw.py and libssw.so files)
        seq1 (str): First sequence to align (one-letter code)
        seq2 (str): Second sequence to align (one-letter code)
        match_point (int, optional): Score awarded for matching characters. Defaults to 2.
        mismatch_penalty (int, optional): Penalty for mismatched characters. Defaults to 2.
        gap_open_penalty (int, optional): Penalty for opening a gap in alignment. Defaults to 3.
        gap_ext_penalty (int, optional): Penalty for extending an existing gap. Defaults to 1.
        protein (bool, optional): If True, treats sequences as protein (amino acids).
                                  If False, treats as nucleotide sequences. Defaults to True.

    Returns:
        float: Smith-Waterman similarity score between the two sequences

    Note:
        Uses the Complete Striped Smith-Waterman Library by Mengyao Zhao:
        https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library
    """

    # 임시 파일 생성 (서열 저장용)
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".fasta"
    ) as target_file:
        target_file.write(f">target\n{seq1}\n")
        target_path = target_file.name

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".fasta"
    ) as query_file:
        query_file.write(f">query\n{seq2}\n")
        query_path = query_file.name

    pyssw_path = os.path.join(src_path, "pyssw.py")
    lib_path = os.path.join(src_path, "libssw.so")
    if not os.path.exists(pyssw_path) or not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Whether {pyssw_path} or {lib_path} file was not found"
        )

    try:
        # pyssw.py 실행 명령어 구성
        cmd = [
            "python2",
            pyssw_path,
            "-l",
            lib_path,
            "-m",
            str(match_point),
            "-x",
            str(mismatch_penalty),
            "-o",
            str(gap_open_penalty),
            "-e",
            str(gap_ext_penalty),
        ]

        if protein:
            cmd.append("-p")

        cmd.extend([target_path, query_path])

        # 명령어 실행 및 출력 캡처
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout

        # 출력에서 점수 추출
        for line in output.split("\n"):
            if "optimal_alignment_score:" in line:
                score_part = line.split("optimal_alignment_score:")[1].strip()
                score = int(score_part.split()[0])
                return score

        # 점수를 찾지 못한 경우
        raise ValueError("Unable to extract alignment score from output")

    finally:
        # 임시 파일 정리
        if os.path.exists(target_path):
            os.unlink(target_path)
        if os.path.exists(query_path):
            os.unlink(query_path)


def calc_protein_multichain_sw(
    pdb_path_1, pdb_path_2, src_path="../smith-waterman", **kwargs
):
    """
    Calculate normalized Smith-Waterman alignment score between two protein pdb files.
    Multi-chain normalized will be applied following Li et al. https://www.sciencedirect.com/science/article/pii/S2405471223001497

    Args:
        pdb_path_1: First pdb file path
        pdb_path_2: Second pdb file path
        src_path: Path to the Smith-Waterman calculator source library compiled for python interface (location of pyssw.py and libssw.so files))
        **kwargs: Additional alignment parameters (match_point, gap_open_penalty, etc.) - see signature of 'calc_smith_waterman_score' function.
    Returns:
        float: Highest normalized alignment score between the chains of heterodimers
    """
    chain_sequences_1 = get_pdb_sequence(
        pdb_path=pdb_path_1, standardize_residue=True, return_one_letter=True
    )
    chain_sequences_2 = get_pdb_sequence(
        pdb_path=pdb_path_2, standardize_residue=True, return_one_letter=True
    )

    pdb_sequences_1 = [seq for _, seq in chain_sequences_1.items()]
    pdb_sequences_2 = [seq for _, seq in chain_sequences_2.items()]

    norm_sw_scores = []

    for i in pdb_sequences_1:
        for j in pdb_sequences_2:
            inter_seq_sw = calc_smith_waterman_score(seq1=i, seq2=j, src_path=src_path)
            intra_seq_sw_1 = calc_smith_waterman_score(
                seq1=i, seq2=i, src_path=src_path
            )
            intra_seq_sw_2 = calc_smith_waterman_score(
                seq1=j, seq2=j, src_path=src_path
            )

            norm_sw_score = inter_seq_sw / np.sqrt(intra_seq_sw_1 * intra_seq_sw_2)
            norm_sw_scores.append(norm_sw_score)

    return round(max(norm_sw_scores), 4)


def prepare_train_val_test_keys(
    tr_vl_pdb_paths,
    test_pdb_paths,
    val_size=0.15,
    seed=312,
    filter_sw_similarity=True,
    sim_threshold=0.7,
    keep_filtered=False,
    sw_lib_dir="../smith-waterman",
    sw_cache_dir="../../cache",
):

    def extract_key_from_pdb_path(pdb_path):
        """Return parent dir name of the pdb file."""
        return os.path.basename(os.path.dirname(pdb_path))

    class SwScoreCache:
        def __init__(self, sw_cache_dir):
            self.sw_cache_dir = sw_cache_dir
            os.makedirs(sw_cache_dir, exist_ok=True)
            self.cache_path = os.path.join(sw_cache_dir, "precalc_sw_scores.pkl")
            self.precalc_sw_scores = {}
            if os.path.exists(self.cache_path):
                try:
                    with open(self.cache_path, "rb") as fp:
                        self.precalc_sw_scores = pickle.load(fp)
                except (pickle.PickleError, IOError, EOFError):
                    print("Warning: Cache file corrupted, starting with empty cache")
            self.modified = False

        def get(self, key_pair):
            return self.precalc_sw_scores.get(key_pair)

        def cache(self, key_pair, sw_score):
            if key_pair not in self.precalc_sw_scores:
                self.precalc_sw_scores[key_pair] = sw_score
                self.modified = True

        def save(self):
            if self.modified:
                with open(self.cache_path, "wb") as fp:
                    pickle.dump(
                        self.precalc_sw_scores, fp, protocol=pickle.HIGHEST_PROTOCOL
                    )
                self.modified = False

    ts_paths = test_pdb_paths.copy()
    ssc = SwScoreCache(sw_cache_dir)

    if filter_sw_similarity:
        filtered_tr_vl_paths = []
        additional_test_paths = []

        for tr_vl_path in tqdm(tr_vl_pdb_paths, desc="Filtering train/val data"):
            should_filter = False
            tr_vl_key = extract_key_from_pdb_path(tr_vl_path)

            for ts_path in test_pdb_paths:
                ts_key = extract_key_from_pdb_path(ts_path)
                key_pair = (ts_key, tr_vl_key)

                # cache에서 먼저 확인
                norm_sw_score = ssc.get(key_pair)

                if norm_sw_score is None:
                    # cache에 없으면 계산
                    norm_sw_score = calc_protein_multichain_sw(
                        ts_path, tr_vl_path, src_path=sw_lib_dir
                    )
                    ssc.cache(key_pair, norm_sw_score)

                if ssc.modified and len(ssc.precalc_sw_scores) % 100 == 0:
                    ssc.save()

                if norm_sw_score > sim_threshold:
                    should_filter = True
                    if keep_filtered:
                        additional_test_paths.append(tr_vl_path)
                    break  # test data중 하나라도 유사하면 비교 종료

            if should_filter == False:
                filtered_tr_vl_paths.append(tr_vl_path)

        ssc.save()

        tr_paths, vl_paths = train_test_split(
            filtered_tr_vl_paths, test_size=val_size, random_state=seed, shuffle=True
        )

        if keep_filtered:
            ts_paths.extend(additional_test_paths)

    else:
        tr_paths, vl_paths = train_test_split(
            tr_vl_pdb_paths, test_size=val_size, random_state=seed, shuffle=True
        )

    tr_keys = [extract_key_from_pdb_path(f) for f in tr_paths]
    vl_keys = [extract_key_from_pdb_path(f) for f in vl_paths]
    ts_keys = [extract_key_from_pdb_path(f) for f in ts_paths]

    return tr_keys, vl_keys, ts_keys


def main():
    """Main function to split data according to specified arguments."""
    args = parse_arguments()
    json_path = os.path.join(args.save_dir, get_file_name(args))
    print(
        f"*********************************\n저장 경로: {json_path}\n*********************************"
    )

    train_pdb_dir, train_vox_dir, test_pdb_dir, test_vox_dir, train_lig_dir, test_lig_dir = get_train_test_dir(args)
    
    total_keys = [
        key
        for key in sorted(os.listdir(train_pdb_dir))
        if os.path.isdir(os.path.join(train_pdb_dir, key))
    ]
    test_keys = read_test_keys(args.test_key_file_path)
    print(
        f"{len(set(total_keys).intersection(set(test_keys)))} overlapping keys between train keys and test keys were found and will be excluded from train keys."
    )
    tr_vl_keys = [key for key in total_keys if key not in test_keys]

    # pdb path extraction
    tr_vl_pdb_paths = []
    for pdb_key in tr_vl_keys:
        tr_vl_pdb_paths.append(get_pdb_path(pdb_key=pdb_key, pdb_dir=train_pdb_dir))

    test_pdb_paths = []
    for pdb_key in test_keys:
        test_pdb_paths.append(get_pdb_path(pdb_key=pdb_key, pdb_dir=test_pdb_dir))

    # configure json file
    data_config_json = {}

    data_config_json["created_at"] = datetime.now().strftime("%Y-%m-%d")
    data_config_json["seed"] = args.seed
    data_config_json["voxel_size"] = args.voxel_size
    data_config_json["input_dim"] = args.input_dim
    data_config_json["index_dir"] = args.index_dir
    data_config_json["lig_file_format"] = args.lig_file_format

    data_config_json["train_source_name"] = args.train_source_name
    data_config_json["train_pdb_dir"] = train_pdb_dir
    data_config_json["train_vox_dir"] = train_vox_dir
    data_config_json["train_lig_dir"] = train_lig_dir

    data_config_json["test_source_name"] = args.test_source_name
    data_config_json["test_pdb_dir"] = test_pdb_dir
    data_config_json["test_vox_dir"] = test_vox_dir
    data_config_json["test_lig_dir"] = test_lig_dir
    data_config_json["smi_emb_dir"] = args.smi_emb_dir
    data_config_json["val_size"] = args.val_size
    data_config_json["tr_data_keys"] = None
    data_config_json["vl_data_keys"] = None
    data_config_json["ts_data_keys"] = None

    data_config_json["filter_sw_similarity"] = args.filter_sw_similarity

    sw_thr = args.sim_threshold if args.filter_sw_similarity else False
    data_config_json["sim_threshold"] = sw_thr
    data_config_json["keep_filtered"] = args.keep_filtered

    tr_data_keys, vl_data_keys, ts_data_keys = prepare_train_val_test_keys(
        tr_vl_pdb_paths=tr_vl_pdb_paths,
        test_pdb_paths=test_pdb_paths,
        val_size=args.val_size,
        seed=args.seed,
        filter_sw_similarity=args.filter_sw_similarity,
        sim_threshold=sw_thr,
        keep_filtered=args.keep_filtered,
        sw_lib_dir=args.sw_lib_dir,
        sw_cache_dir=args.sw_cache_dir,
    )

    data_config_json["tr_data_keys"] = tr_data_keys
    data_config_json["vl_data_keys"] = vl_data_keys
    data_config_json["ts_data_keys"] = ts_data_keys

    # 파일 저장하고 return
    with open(json_path, "w") as fp:
        json.dump(data_config_json, fp, ensure_ascii=False, indent=4)

    return data_config_json


if __name__ == "__main__":
    main()

    # print(get_file_name(args))
    # print(read_test_keys(args.test_key_file_path))
    # print(get_pdb_path('1a2b_1', args.scPDB_pdb_dir))
    # print(get_pdb_sequence("/home/tech/Hawon/Develop/Interaction_free/MAIN/seq2str/pdb_entry_from_dav_kib/8P5G.pdb"))
    # print(calc_sw_similarity2(src_path="../smith-waterman",
    #  seq1="MATVPELKCFQGVDEK",
    #  seq2="MSSVPELKCTQGVDER"))

    # print(
    #     get_pdb_sequence(
    #         pdb_path="/home/tech/Hawon/Develop/Interaction_free/MAIN/seq2str/pdb_entry_from_dav_kib/7SHQ.pdb",
    #         standardize_residue=True,
    #         return_one_letter=True
    #     )
    # )

    # # 예시 아미노산 서열
    # seq1 = "MSSVPELKCTQGVDER"
    # seq2 = "MSSVPELKCTQGVDER"

    # # 전체 정렬 정보 출력
    # result = calc_smith_waterman_score(
    #     src_path="../smith-waterman",
    #     seq1=seq1,
    #     seq2=seq2,
    #     protein=True
    # )

    # print(result)

    # print(calc_protein_multichain_sw("/home/tech/Hawon/Develop/Interaction_free/MAIN/seq2str/pdb_entry_from_dav_kib/8P5G.pdb",
    #                                 "/home/tech/Hawon/Develop/Interaction_free/MAIN/seq2str/pdb_entry_from_dav_kib/7SHQ.pdb"))