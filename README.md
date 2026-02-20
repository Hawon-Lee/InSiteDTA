# InSiteDTA
<img width="4200" height="1498" alt="Fig_overview" src="https://github.com/user-attachments/assets/66a2831e-2014-44ad-be55-5fe7f8ed609f" />

A complex-free deep learning model for protein-ligand binding affinity prediction with intrinsic binding site detection.

**Key Features:**
- No molecular docking required
- No explicit binding site annotation needed
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

**Our tested environment:**
- Python: 3.9.19
- PyTorch: 2.5.1
- PyTorch Geometric: 2.6.1
- CUDA: 11.8

## Quick Start

```bash
python 01-inference.py \
    --pdb_path <path_to_protein.pdb> \
    --smiles <ligand_smiles>
```

**Example:**
```bash
python 01-inference.py \
    --pdb_path ./src/data/sample/1bzc_protein.pdb \
    --smiles "[O-]C(=O)CC[C@@H](C(=O)N)NC(=O)c1ccc2c(c1)ccc(c2)C(P(=O)([O-])[O-])(F)F"
```

## Standard Training Procedure
```bash
sh standard_train.sh 를 실행하세요
```

## Training With Your Own Data
```bash
python 02-preprocess.py --(당신은 2개의 raw data 구조를 채택할 수 있습니다.)
하나는 pdbbind-like architecture (ligand structure info 가 있을 경우 권장)
(data/{pdb_id}/{pdb_id}_protein.pdb)

다른 하나는 flatten architecture (ligand structure info 가 없고 smiles 만 있을 경우 권장)
(data/proteins/{*pdb_id}_protein.pdb)
```

// [TODO]: 만약 binding site 만 예측하고 싶다면 -> index file 을 None 으로 주세요

## Reproduce Paper Results

Run evaluation on three benchmark datasets:
```bash
# Evaluate on Coreset_crystal
python 04-reproduce.py --data crystal --batch_size 64 --device 0

# Evaluate on Coreset_redocked  
python 04-reproduce.py --data redocked --batch_size 64 --device 0

# Evaluate on Coreset_p2rank
python 04-reproduce.py --data p2rank --batch_size 64 --device 0
```

The script will:
1. Prepare ligand features from SMILES
2. Voxelize protein structures
3. Evaluate with three trained models
4. Report performance metrics (PCC, RMSE, MAE)

## Output

The model outputs predicted binding affinity in pK scale (higher values indicate stronger binding).

## Data

**$Coreset_{crystal}$**
- Standard benchmark dataset from PDBbind

**$Coreset_{redocked}$**
- Coreset with redocked ligand in the native pocket

**$Coreset_{p2rank}$**
- Coreset with redocked ligand in the p2rank predicted pocket


## Citation

TBD
