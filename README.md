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

### 3. Install P2Rank (Optional, Recommended)
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

### Step 3: Train

```bash
python 03-train.py \
    --data_config ./preprocessed/data_config_*.json \
    --save_dir ./checkpoints \
    --gpu 0 \
    --epochs 300 \
    --batch_size 48
```

Trained model will be saved as `./checkpoints/{timestamp}.pt`.

## Reproduce Paper Results

Run evaluation on three benchmark datasets:
```bash
# Evaluate on Coreset_crystal
python 05-reproduce.py --data crystal --batch_size 64 --device 0

# Evaluate on Coreset_redocked  
python 05-reproduce.py --data redocked --batch_size 64 --device 0

# Evaluate on Coreset_p2rank
python 05-reproduce.py --data p2rank --batch_size 64 --device 0
```

The script will:
1. Prepare ligand features from SMILES
2. Voxelize protein structures
3. Evaluate with three trained models
4. Report performance metrics (PCC, RMSE, MAE)

## Output

**Inference (01-inference.py):**
- Predicted binding affinity in pK scale (higher values = stronger binding)

**Training (03-train.py):**
- Model checkpoint: `{save_dir}/{timestamp}.pt`
- Training results: `{save_dir}/{timestamp}_results.json`

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
