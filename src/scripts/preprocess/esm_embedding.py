import os, argparse, pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import warnings
from Bio import BiopythonWarning
warnings.simplefilter("ignore", BiopythonWarning)


AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'C', 'SEP': 'S', 'TPO': 'T', 'PTR': 'Y',
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
}


def parse_pdb_chains(pdb_path: str) -> dict:
    """
    PDB 파일에서 체인별 서열과 CA 좌표를 추출한다.

    Returns:
        {chain_id: {"sequence": str, "ca_coords": np.array (N, 3)}}
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    chains = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            sequence = []
            ca_coords = []

            for residue in chain:
                if residue.get_id()[0] != ' ':
                    continue

                resname = residue.get_resname().strip()
                aa = AA_MAP.get(resname, 'X')

                if 'CA' in residue:
                    sequence.append(aa)
                    ca_coords.append(residue['CA'].get_vector().get_array())

            if sequence:
                chains[chain_id] = {
                    "sequence": "".join(sequence),
                    "ca_coords": np.array(ca_coords, dtype=np.float32),
                }
        break  # first model only

    return chains


def extract_esm_embeddings(pdb_path: str, model, device="cuda:0") -> dict:
    """
    PDB 파일 하나로부터 ESM-C 잔기별 embedding을 추출한다.
    멀티체인은 체인별로 개별 추론하며, homo-oligomer는 한 번만 추론한다.

    Returns:
        {"ca_coords": np.array (N_total, 3), "embeddings": np.array (N_total, D)}
    """
    from esm.sdk.api import ESMProtein, LogitsConfig

    chains = parse_pdb_chains(pdb_path)
    if not chains:
        return {
            "ca_coords": np.zeros((0, 3), dtype=np.float32),
            "embeddings": np.zeros((0, 960), dtype=np.float32),
        }

    # Group chains by identical sequence (homo-oligomer optimization)
    seq_to_chains = defaultdict(list)
    for chain_id, data in chains.items():
        seq_to_chains[data["sequence"]].append(chain_id)

    # Run ESM-C once per unique sequence
    seq_embeddings = {}
    for seq in seq_to_chains:
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
        # Remove BOS/EOS tokens: (1, seq_len+2, D) -> (seq_len, D)
        emb = output.embeddings[0, 1:-1, :].detach().cpu().numpy()
        seq_embeddings[seq] = emb

    # Assemble all chains
    all_ca_coords = []
    all_embeddings = []

    for chain_id, data in chains.items():
        seq = data["sequence"]
        emb = seq_embeddings[seq]
        ca = data["ca_coords"]

        # Length mismatch guard (rare: missing CA atoms)
        min_len = min(len(emb), len(ca))
        all_ca_coords.append(ca[:min_len])
        all_embeddings.append(emb[:min_len])

    return {
        "ca_coords": np.concatenate(all_ca_coords, axis=0).astype(np.float32),
        "embeddings": np.concatenate(all_embeddings, axis=0).astype(np.float32),
    }


def batch_extract_esm(pdb_dir: str, save_dir: str, device="cuda:0",
                       data_structure="nested"):
    """
    디렉토리 내 전체 PDB에 대해 ESM-C embedding을 추출하여 저장한다.

    Args:
        pdb_dir: PDB 파일이 있는 디렉토리
        save_dir: ESM pkl 파일 저장 디렉토리
        device: CUDA 디바이스 문자열
        data_structure: "nested" (pdbbind 형식) or "flatten"
    """
    from esm.models.esmc import ESMC

    os.makedirs(save_dir, exist_ok=True)

    model = ESMC.from_pretrained("esmc_300m").to(device)
    model.eval()

    if data_structure == "nested":
        pdb_ids = sorted(
            [d for d in os.listdir(pdb_dir)
             if os.path.isdir(os.path.join(pdb_dir, d))]
        )
    else:
        raise NotImplementedError("flatten mode not supported yet")

    for pdb_id in tqdm(pdb_ids, desc="Extracting ESM-C embeddings"):
        out_path = os.path.join(save_dir, f"{pdb_id}_esm.pkl")
        if os.path.exists(out_path):
            continue

        pdb_path = os.path.join(pdb_dir, pdb_id, f"{pdb_id}_protein.pdb")
        if not os.path.exists(pdb_path):
            print(f"[Warning] PDB not found: {pdb_path}")
            continue

        try:
            result = extract_esm_embeddings(pdb_path, model, device)
            with open(out_path, "wb") as fp:
                pickle.dump(result, fp)
        except Exception as e:
            print(f"[Error] {pdb_id}: {e}")
            continue


def fit_pca(esm_dir: str, n_components: int = 64, save_path: str = None):
    """
    추출된 ESM embedding 전체에 대해 PCA를 fitting 한다.
    sklearn 없이도 transform 가능하도록 mean/components를 dict로 저장한다.
    """
    from sklearn.decomposition import PCA

    if save_path is None:
        save_path = os.path.join(esm_dir, f"pca_{n_components}.pkl")

    if os.path.exists(save_path):
        print(f"PCA model already exists: {save_path}")
        return

    all_embeddings = []
    esm_files = sorted([f for f in os.listdir(esm_dir) if f.endswith("_esm.pkl")])

    for f in tqdm(esm_files, desc="Loading embeddings for PCA"):
        with open(os.path.join(esm_dir, f), "rb") as fp:
            data = pickle.load(fp)
            if len(data["embeddings"]) > 0:
                all_embeddings.append(data["embeddings"])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Fitting PCA: {all_embeddings.shape[0]} residues, "
          f"{all_embeddings.shape[1]} -> {n_components} dims")

    pca = PCA(n_components=n_components)
    pca.fit(all_embeddings)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Explained variance ratio: {explained:.4f}")

    pca_data = {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_components": n_components,
    }
    with open(save_path, "wb") as fp:
        pickle.dump(pca_data, fp)

    print(f"PCA model saved: {save_path}")


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Extract ESM-C per-residue embeddings from PDB files"
    )
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Directory containing PDB subdirectories")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for ESM pkl files")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data_structure", type=str, default="nested",
                        choices=["nested", "flatten"])
    parser.add_argument("--fit_pca", action="store_true",
                        help="Fit PCA after extraction")
    parser.add_argument("--pca_dim", type=int, default=64,
                        help="Number of PCA components")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    batch_extract_esm(
        pdb_dir=args.pdb_dir,
        save_dir=args.save_dir,
        device=f"cuda:{args.device}",
        data_structure=args.data_structure,
    )

    if args.fit_pca:
        fit_pca(esm_dir=args.save_dir, n_components=args.pca_dim)
