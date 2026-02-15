import os, json, sys, random
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from .dataset import CustomDataset

sys.path.append("../")

def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def collate_as_dict(items: list) -> dict[str, any]:
    """
    Efficiently collates a list of dictionaries into batched tensors and objects.
    
    Args:
        items: List of dictionaries containing arrays, tensors, strings, or Data objects
    Returns:
        Dictionary mapping keys to batched tensors or Batch objects
    """
    if not items:
        return {}
    
    # Pre-compute batch size and get reference keys from first item
    batch_size = len(items)
    reference_item = items[0]
    
    # Initialize result containers for each key based on first item's types
    batches = {}
    for key, value in reference_item.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            # Pre-allocate maximum sized array by checking all items
            max_dims = [value.shape]
            for item in items[1:]:
                if key in item and isinstance(item[key], (np.ndarray, torch.Tensor)):
                    max_dims.append(item[key].shape)
            max_shape = np.maximum.reduce(max_dims)
            batches[key] = np.zeros((batch_size, *max_shape))
            
        elif isinstance(value, str):
            batches[key] = ["" for _ in range(batch_size)]
            
        elif isinstance(value, Data):
            # Create list to collect Data objects for batch conversion
            batches[key] = []
            
        elif isinstance(value, float):
            batches[key] = torch.zeros(batch_size, dtype=torch.float32)
            
        else:
            batches[key] = np.zeros(batch_size)

    # Single pass to fill batches
    for batch_idx, item in enumerate(items):
        if item is None:
            continue
            
        for key, value in item.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                # Efficient slicing using pre-computed shapes
                slices = tuple(slice(0, dim) for dim in value.shape)
                batches[key][(batch_idx, *slices)] = value
                
            elif isinstance(value, Data):
                # Collect Data objects for batch conversion
                batches[key].append(value)
            
            elif isinstance(batches[key], torch.Tensor):
                batches[key][batch_idx] = torch.tensor(value, dtype=torch.float32)
                
            else:
                batches[key][batch_idx] = value

    # Post-process batches
    result = {}
    for key, batch in batches.items():
        if isinstance(reference_item[key], np.ndarray):
            # Convert numpy arrays to PyTorch tensors
            result[key] = torch.from_numpy(batch).float()
        elif isinstance(reference_item[key], Data):
            # Convert collected Data objects to Batch
            result[key] = Batch.from_data_list(batch)
        elif isinstance(batch, np.ndarray):
            result[key] = torch.from_numpy(batch).float()
        else:
            result[key] = batch

    return result


class MasterDataLoader:
    def __init__(self, data_config, train_config):
        """
        Args:
            data_config (dict)
            train_config (namespace)
        """
        
        # From data_config
        self.seed = data_config["seed"]
        self.lig_file_format = data_config["lig_file_format"]

        self.train_source_name = data_config["train_source_name"]
        self.train_lig_dir = data_config["train_lig_dir"]
        self.train_vox_dir = data_config["train_vox_dir"]
        
        self.test_source_name = data_config["test_source_name"]
        self.test_lig_dir = data_config["test_lig_dir"]
        self.test_vox_dir = data_config["test_vox_dir"]
        
        self.index_dir = data_config["index_dir"]
        self.train_index_dict = self.read_index_file(self.train_source_name)
        self.test_index_dict = self.read_index_file(self.test_source_name)

        self.tr_data_keys = data_config["tr_data_keys"]
        self.vl_data_keys = data_config["vl_data_keys"]
        self.ts_data_keys = data_config["ts_data_keys"]
        
        self.filter_sw_similarity = data_config["filter_sw_similarity"]
        self.sim_threshold = data_config["sim_threshold"]
        self.keep_filtered = data_config["keep_filtered"]
            
        self.batch_size = train_config.batch_size
        self.num_workers = train_config.num_workers
        
    def read_index_file(self, dataset_name):
        index_file = {}
        
        if dataset_name == "PDBbind2020":
            index_path = os.path.join(self.index_dir, "PDBbind_aff_index.json")
        elif dataset_name in ("Davis", "Filtered_Davis"):
            index_path = os.path.join(self.index_dir, "Davis_aff_index.json")
        elif dataset_name == "Kiba":
            index_path = os.path.join(self.index_dir, "Kiba_aff_index.json")
        
        try:
            with open(index_path, 'r') as fp:
                index_file = json.load(fp)
        except:
            print("No binding affinity index file was found. All binding affinity values will be set to NaN.")
            return {}
        
        return index_file

    def get_vox_lig_zip_paths(self, vox_dir, lig_dir, keys, lig_file_format="sdf", desc="training"):
        """
        Match voxel files and ligand files based on keys and return a zipped list
        """
        from pathlib import Path
        from collections import defaultdict
        
        vox_lig_pairs = []
        is_nested = True
        
        if lig_file_format == "rdmol":
            lig_file_format = "pkl"
            is_nested = False
        
        # 1. Scan all voxel files at once and group by key
        print(f"Scanning {desc} voxel files in {vox_dir}...")
        vox_files_by_key = defaultdict(list)
        try:
            with os.scandir(vox_dir) as entries:
                for entry in entries:
                    if entry.is_file() and "dim" in entry.name:
                        for key in keys:
                            if key in entry.name:
                                vox_files_by_key[key].append(entry.path)
                                break  # Use only the first matching key
        except (PermissionError, OSError) as e:
            print(f"Error scanning vox_dir: {e}")
            return []
        
        # 2. Efficiently find ligand files
        print(f"Scanning {desc} ligand files in {lig_dir}...")
        lig_files_by_key = {}
        
        if is_nested:
            # nested structure: find files in each key directory
            lig_base_path = Path(lig_dir)
            for key in keys:
                key_dir = lig_base_path / key
                if key_dir.exists() and key_dir.is_dir():
                    lig_files = list(key_dir.glob(f"*.{lig_file_format}"))
                    if lig_files:
                        lig_files_by_key[key] = [str(f) for f in sorted(lig_files)]
        else:
            # flat structure: find directly by filename
            lig_base_path = Path(lig_dir)
            if lig_base_path.exists():
                for key in keys:
                    lig_file = lig_base_path / f"{key}_ligand.{lig_file_format}"
                    if lig_file.exists():
                        lig_files_by_key[key] = [str(lig_file)]
        
        # 3. Match and create pairs
        print(f"Matching {desc} vox-lig pairs...")
        matched_keys = 0
        
        for key in keys:
            vox_files = sorted(vox_files_by_key.get(key, []))
            lig_files = lig_files_by_key.get(key, [])
            
            if not vox_files or not lig_files:
                continue
                
            # Matching strategy based on file count
            if len(vox_files) == len(lig_files): # 1:1 matching
                vox_lig_pairs.extend(zip(vox_files, lig_files))
                matched_keys += 1
            elif len(vox_files) == 1: # 1 voxel -> match with first ligand
                vox_lig_pairs.append((vox_files[0], lig_files[0]))
                matched_keys += 1
            elif len(lig_files) == 1: # 1 ligand -> match with first voxel
                vox_lig_pairs.append((vox_files[0], lig_files[0]))
                matched_keys += 1
        
        print(f"Found {len(vox_lig_pairs)} pairs from {matched_keys}/{len(keys)} keys")
        print(f"- Samples: Lig {len(lig_files_by_key)}, Ptn {len(vox_files_by_key)} \n")
        return vox_lig_pairs    
    
    def get_vox_file_paths(self, vox_dir, keys, desc="training"):
        vox_file_paths = []
        vox_file_list = os.listdir(vox_dir)
        
        for f in tqdm(vox_file_list, desc=f"Searching {desc} voxel data..."):
            if "dim" in f and any(key in f for key in keys):
                vox_file_paths.append(os.path.join(vox_dir, f))

        return sorted(vox_file_paths)
        
    def get_lig_file_paths(self, lig_dir, keys, lig_file_format="sdf", desc="training"):
        lig_file_paths = []
        is_nested = True
        
        if lig_file_format == "rdmol":
            lig_file_format = "pkl"
            is_nested = False
        
        if is_nested:
            patterns = [os.path.join(lig_dir, k, f"*.{lig_file_format}") for k in keys]
        else:
            patterns = [f"{lig_dir}/{k}_ligand.{lig_file_format}" for k in keys]
            
        for pattern in tqdm(patterns, desc=f"Searching {desc} voxel data..."):
            lig_file_paths.extend(glob(pattern))
        
        return sorted(lig_file_paths)

    def get_tr_vl_dataset(self):
        tr_zip_paths = self.get_vox_lig_zip_paths(self.train_vox_dir, self.train_lig_dir, self.tr_data_keys, self.lig_file_format, desc="training")
        vl_zip_paths = self.get_vox_lig_zip_paths(self.train_vox_dir, self.train_lig_dir, self.vl_data_keys, self.lig_file_format, desc="validation")
        
        tr_dataset = CustomDataset(data_zip_paths=tr_zip_paths, index_dict=self.train_index_dict)
        vl_dataset = CustomDataset(data_zip_paths=vl_zip_paths, index_dict=self.train_index_dict)
        return tr_dataset, vl_dataset
    
    def get_tr_vl_dataloader(self):
        tr_dataset, vl_dataset = self.get_tr_vl_dataset()
        
        tr_g = seed_generator(self.seed)
        vl_g = seed_generator(self.seed + 1)
        
        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_as_dict,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=tr_g,
            pin_memory=True,
            shuffle=True)
        
        vl_dataloader = DataLoader(
            vl_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_as_dict,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=vl_g,
            pin_memory=True,
            shuffle=False)
        
        return tr_dataloader, vl_dataloader
    
    def get_ts_dataset(self):
        ts_zip_paths = self.get_vox_lig_zip_paths(self.test_vox_dir, self.test_lig_dir, self.ts_data_keys, self.lig_file_format, desc="test")

        # ts_vox_files = self.get_vox_file_paths(self.test_vox_dir, self.ts_data_keys, desc="test")
        # ts_lig_files = self.get_lig_file_paths(self.test_lig_dir, self.ts_data_keys, lig_file_format=self.lig_file_format, desc="test")
        # # protein-ligand file key matching이 잘 되었는지 점검
        # if not self.check_key_matches(ts_vox_files, ts_lig_files):
        #     raise ValueError("Mismatched keys found between voxel and ligand files.")
        # ts_zip_paths = list(zip(ts_vox_files, ts_lig_files))
        
        ts_dataset = CustomDataset(data_zip_paths=ts_zip_paths, index_dict=self.test_index_dict)
        return ts_dataset
    
    def get_ts_dataloader(self):
        ts_dataset = self.get_ts_dataset()
        
        ts_g = seed_generator(self.seed + 2)
        
        ts_dataloader = DataLoader(
            ts_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_as_dict,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=ts_g,
            pin_memory=True,
            shuffle=False)
        
        return ts_dataloader
    
    def get_dataloader_from_dataset(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_as_dict,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            shuffle=False
        )

        return dataloader