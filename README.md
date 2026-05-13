# InSiteDTA
<img width="4200" height="1498" alt="Fig_overview" src="https://github.com/user-attachments/assets/66a2831e-2014-44ad-be55-5fe7f8ed609f" />

A complex-free deep learning model for protein-ligand binding affinity prediction with intrinsic binding site detection.

**Key Features:**
- No molecular docking required
- Robust performance regardless of binding site determination method
- Robust performance on imperfect structural inputs

## Installation

### 1. Clone repository
```bash
git clone https://github.com/KU-MedAI/InSiteDTA.git
cd InSiteDTA
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate insite
```

### 3. Install P2Rank (Optional, Recommended) — [Krivák & Hoksza, 2018](https://doi.org/10.1186/s13321-018-0285-8)
```bash
mkdir src/p2rank && cd src/p2rank
wget https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz
tar -xzf p2rank_2.5.1.tar.gz -C ./ --strip-components=1
```

> **Why P2Rank?**
> InSiteDTA internally predicts the binding site and uses it as a feature for affinity prediction, so P2Rank is not strictly required. However, providing a P2Rank-predicted pocket helps guide the voxelization step so that the sampled protein voxel is more likely to include the true binding site. This can enable more sophisticated prediction, especially when inferencing with large proteins.

**Our tested environment:**
- Python: 3.9.19
- PyTorch: 2.5.1
- PyTorch Geometric: 2.6.1
- CUDA: 11.8
- P2Rank: 2.5.1

## Quick Start Example

**Without pocket guidance (unguided voxelization):**
```bash
python 01-inference.py \
    --pdb_path ./src/data/samples/4gkm/4gkm_protein.pdb \
    --smiles "Cc1ccc(c(c1)C(=O)[O-])Nc1ccccc1C(=O)[O-]"
```

**With P2Rank guidance (guided voxelization, recommended):**
```bash
python 01-inference.py \
    --pdb_path ./src/data/samples/4gkm/4gkm_protein.pdb \
    --smiles "Cc1ccc(c(c1)C(=O)[O-])Nc1ccccc1C(=O)[O-]" \
    --use_p2rank
```

## Training With Your Own Data

### Step 1: Prepare Data Structure

Organize your data in nested structure (PDBbind format):
```
raw_data/
├── {pdb_id}/
│   ├── {pdb_id}_protein.pdb
│   └── {pdb_id}_pocket.pdb
...
```

Prepare SMILES CSV file (`smiles.csv`):
```csv
PDB_ID,Canonical SMILES
1abc,CCO
1def,c1ccccc1
```

For affinity prediction, prepare affinity index JSON (`affinity.json`):
```json
{"1abc": 5.2, "1def": 7.8}
```
> **Note:** If you only want to train binding site prediction, omit `--index_file` argument in preprocessing.

### Step 2: Preprocess

```bash
python 02-preprocess.py \
    --raw_dir ./raw_data \
    --save_dir ./preprocessed \
    --smiles_csv ./smiles.csv \
    --index_file ./affinity.json \
    --test_key_file ./test_keys.txt \
    --voxel_size 2 \
    --n_voxels 32 \
    --device 0
```

This generates preprocessed data and `data_config_*.json` in `./preprocessed/`.

//TODO: Integrated venvs
#### Optional: ESM-C Protein Embeddings

You can enrich protein voxels with per-residue ESM-C embeddings (960-dim, mapped to CA coordinates via Gaussian density).

**Step 2a: Extract ESM-C embeddings** (requires `esm` environment)
```bash
# Create environment
conda create -n esm python=3.10 -y
conda run -n esm pip install esm biopython httpx
conda run -n esm pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Extract embeddings and fit PCA
conda run -n esm python src/scripts/preprocess/esm_embedding.py \
    --pdb_dir ./raw_data \
    --save_dir ./preprocessed/esm_embeddings \
    --device 0 \
    --fit_pca --pca_dim 64
```

This saves full 960-dim embeddings (`{pdb_id}_esm.pkl`) and a PCA model (`pca_64.pkl`).
Multi-chain proteins are handled by processing each chain separately; homo-oligomers are optimized to run ESM inference only once per unique sequence.

**Step 2b: Voxelize with ESM features** (requires `insite` environment)
```bash
python 02-preprocess.py \
    --raw_dir ./raw_data \
    --save_dir ./preprocessed \
    --smiles_csv ./smiles.csv \
    --index_file ./affinity.json \
    --test_key_file ./test_keys.txt \
    --voxel_size 2 \
    --n_voxels 32 \
    --device 0 \
    --esm_dir ./preprocessed/esm_embeddings \
    --pca_path ./preprocessed/esm_embeddings/pca_64.pkl
```

This produces voxels with 21 (hand-crafted) + 64 (ESM PCA) + 1 (pocket label) = 86 channels.
Set `in_channels=85` when creating the model. If `--pca_path` is omitted, full 960-dim ESM features are used.

### Step 3: Train

```bash
python 03-train.py \
    --data_config ./preprocessed/data_config_*.json \
    --save_dir ./checkpoints \
    --device 0 \
    --epochs 300 \
    --batch_size 48
```

Trained model will be saved as `./checkpoints/{timestamp}_{data_config_name}.pt`.

## Evaluate Your Trained Model

```bash
python 04-evaluate.py \
    --ckpt ./checkpoints/{experiment_name}.pt \
    --result_file ./checkpoints/{experiment_name}_results.json \
    --save_dir ./evaluation \
    --device 0
```

The script will:
1. Load the test split defined in the training result file
2. Run inference on the test set
3. Report performance metrics (PCC, RMSE, MAE, DCC, DVO)
4. Save detailed results to `{save_dir}/{experiment_name}_test_results.csv`

## Reproduce Paper Results

Run evaluation on three benchmark datasets:
```bash
# Evaluate on Coreset_crystal
python 05-reproduce.py --ckpt ckpt_CleanSplit_s309_teacher.pt --scenario crystal --device 0

# Evaluate on Coreset_redocked  
python 05-reproduce.py --ckpt ckpt_CleanSplit_s309_teacher.pt --scenario redocked --device 0

# Evaluate on Coreset_p2rank
python 05-reproduce.py --ckpt ckpt_CleanSplit_s309_teacher.pt --scenario p2rank --device 0
```

To evaluate with ESM-enriched voxels, add `--esm_dir` and `--pca_path`:
```bash
python 05-reproduce.py --scenario crystal --batch_size 64 --device 0 \
    --esm_dir ./preprocessed/esm_embeddings \
    --pca_path ./preprocessed/esm_embeddings/pca_64.pkl
```

The script will:
1. Prepare ligand features from SMILES
2. Voxelize protein structures (with optional ESM features)
3. Evaluate with three trained models
4. Report performance metrics (PCC, RMSE, MAE)

## Output

**Inference (01-inference.py):**
- Predicted binding affinity in pK scale (higher values = stronger binding)

**Training (03-train.py):**
- Model checkpoint: `{save_dir}/{timestamp}_{data_config_name}.pt`
- Training results: `{save_dir}/{timestamp}_{data_config_name}_results.json`

**Evaluate (04-evaluate.py):**
- Evaluation results CSV: `{save_dir}/{experiment_name}_test_results.csv`

**Reproduce (05-reproduce.py):**
- Performance metrics (mean ± std across 3 models): PCC, RMSE, MAE

## Data

**$Coreset_{crystal}$**
- Standard benchmark dataset from PDBbind

**$Coreset_{redocked}$**
- Coreset with redocked ligand in the native pocket

**$Coreset_{p2rank}$**
- Coreset with redocked ligand in the p2rank predicted pocket


## Citation

TBD
