import pickle
import os, sys
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  # 여기 중요!! rdkit sdf 읽을때 kekulize error 누르는 부분

def create_mol_from_file(file_path: str, retry_without_sanitize: bool = False) -> Chem.Mol:
    '''
    Create RDKit mol object from input structure file.
    Automatically detect 'sdf, mol2, pdb' file extension.
    
    Args:
        file_path (str): Input file path which ends with file extension of sdf, mol2, pdb and pkl (valid only when containing rdkit.Mol object).
        retry_without_sanitize (bool): Whether to retry with sanitize=False when initial creation returns none
    
    Returns:
        Chem.Mol: Created RDKit molecule object
    '''
    
    def load_pickle(path):
        with open(path, 'rb') as fp:
            return pickle.load(fp)
    
    # Mapping of file extensions to reader functions
    file_readers = {
        'sdf':Chem.MolFromMolFile,
        'mol2':Chem.MolFromMol2File,
        'pdb':Chem.MolFromPDBFile,
        'pkl':load_pickle
    }
    
    # Extract file extension and map to appropriate reader function
    file_extension = file_path.split('.')[-1].lower()
    reader_func = file_readers.get(file_extension)
    
    # Raise error when input file extension is not supported.
    if reader_func is None:
        raise ValueError(f"Unsupported file format : {file_extension}")
    
    # Create mol
    m = reader_func(file_path)
    # Retry with sanitize=False if option is set and initial attempt failed
    if m is None:
        if file_extension != "pkl" and retry_without_sanitize:
            m = reader_func(file_path, sanitize=False)

    return m

def remove_hydrogens(mol):
    rm_hydrogen_methods = [
        lambda m: Chem.RemoveHs(m),
        lambda m: Chem.RemoveHs(m, implicitOnly=True),
        lambda m: Chem.RemoveHs(m, implicitOnly=True, sanitize=False) # 점점 관대한 방식 적용
    ]
    
    for method in rm_hydrogen_methods:
        try:
            return method(mol)
        except (ValueError, RuntimeError, Exception) as e:
            continue
    
    # 모든 방법 실패
    return None

def check_usable_lig(lig_path):
    """
    리간드 파일에서 유효한 분자 객체를 생성할 수 있는지 검증합니다.
    
    Args:
        lig_path (str): 리간드 파일 경로
        lig_file_format (str): 리간드 파일의 확장자 (sdf, mol2, rdmol)
        
    Returns:
        bool: 유효한 분자 객체 생성 가능 여부
    """
    
    try:
        # 1. 분자 객체 생성
        mol = create_mol_from_file(lig_path, retry_without_sanitize=True)
        if mol is None:
            return False
        
        # 2. 분자 기본 검증 (원자 수 확인)
        if mol.GetNumAtoms() == 0:
            return False
        
        # 3. 수소 제거 - 세 가지 방법 중 하나를 선택적으로 적용
        mol = remove_hydrogens(mol)
        if mol is None:
            return False
        
        return True
        
    except Exception:
        return False
    
def encode_ligand_to_Data(mol: Chem.rdchem.Mol):    
    
    lpp = LigandPreprocessor(mol)
    
    # 1. generate ligand_feature (54dim)
    lig_feature = lpp.get_lig_feature(mol, to_tensor=True)
    
    # 2. generate adj_matrix
    lig_edge_indices, lig_edge_attr = lpp.get_edge_info(mol)

    # pyg style adj가 아니라 numpy style adj (nxn) 인 경우 tensor로 변환
    if isinstance(lig_edge_indices, np.ndarray):
        lig_edge_indices = torch.from_numpy(lig_edge_indices)
    
    # 3. generate atom_pos
    lig_pos = lpp.get_atom_position(mol)
    
    # 4. generate atomic_number z
    lig_z = lpp.get_atomic_number(mol)
    
    # 최종 Data객체 생성    
    ligand_data = Data(x=lig_feature, edge_index=lig_edge_indices, edge_attr=lig_edge_attr, pos=lig_pos, z=lig_z)
    
    return ligand_data
    
    
class LigandPreprocessor:
    def __init__(self, mol, num_conformers=5):
        self.mol = mol
        self.num_conformers = num_conformers
        
        self.SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "X"]
        self.DEGREES = [0, 1, 2, 3, 4, 5]
        self.HYBRIDIZATIONS = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ]
        self.FORMALCHARGES = [-2, -1, 0, 1, 2, 3, 4]
        
        pt = """
            H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
            LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
            NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
            K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
            RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
            CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
        """
        
        self.PERIODIC_TABLE = dict()
        for i, per in enumerate(pt.split()):
            for j, ele in enumerate(per.split(",")):
                self.PERIODIC_TABLE[ele] = (i, j)
        self.PERIODS = [0, 1, 2, 3, 4, 5]
        self.GROUPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        
    def one_of_k_encoding(self, x, allowable_set: list) -> list[bool]:
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set: list) -> list[bool]:
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
    
    def get_period_group(self, atom) -> list[bool]:
        period, group = self.PERIODIC_TABLE[atom.GetSymbol().upper()]
        return self.one_of_k_encoding(period, self.PERIODS) + self.one_of_k_encoding(group, self.GROUPS)
    
    def atom_feature(self, mol, atom_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        return np.array(
            self.one_of_k_encoding_unk(atom.GetSymbol(), self.SYMBOLS)
            + self.one_of_k_encoding_unk(atom.GetDegree(), self.DEGREES)
            + self.one_of_k_encoding_unk(atom.GetHybridization(), self.HYBRIDIZATIONS)
            + self.one_of_k_encoding_unk(atom.GetFormalCharge(), self.FORMALCHARGES)
            + self.get_period_group(atom)
            + [atom.GetIsAromatic()]
        )  # (9, 6, 7, 7, 24, 1) --> total 54        
    
    def get_lig_feature(self, mol, to_tensor=False):
        '''
        mol 객체의 원자를 순회하면서 atom_feature 함수를 활용해 분자 feature 생성
        to_tensor -> return을 numpy 대신 tensor
        '''
        n_atoms = mol.GetNumAtoms()
        atom_features = []
        
        for atom_idx in range(n_atoms):
            atom_features.append(self.atom_feature(mol, atom_idx))
            
        lig_feature = np.array(atom_features)
        
        if to_tensor:
            lig_feature = torch.from_numpy(lig_feature).to(dtype=torch.float32)
            
        return lig_feature
        
    def get_edge_info(self, mol):
        adj = Chem.GetAdjacencyMatrix(mol, useBO=True) # [n_atoms, n_atoms]
        
        edge_index = torch.tensor(np.vstack(adj.nonzero()), dtype=torch.long) # [2, n_edges]
        edge_attr = torch.tensor(adj[adj.nonzero()], dtype=torch.float) # [n_edges, ]
            
        return edge_index, edge_attr

    def get_atom_position(self, mol, to_tensor=True, numConfs=None):
        # 3D conformer 로 변환

        # mol.RemoveAllConformers() # standardize conformation
        # AllChem.EmbedMolecules(mol)
        # AllChem.MMFFOptimizeMolecule(mol)
        # 원하는 conformer 갯수만큼 슬라이싱
        if numConfs is None:
            numConfs = self.num_conformers
            
        pos_list = []
        for n in range(numConfs):
            conf = mol.GetConformer(n)
            pos = conf.GetPositions()
            pos_list.append(pos)
        
        stacked_pos = np.stack(pos_list, axis=1)
            
        if to_tensor:
            stacked_pos = torch.from_numpy(stacked_pos).to(dtype=torch.float32)
        
        return stacked_pos

    def get_atomic_number(self, mol, to_tensor=True):
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        if to_tensor:
            atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
            
        return atomic_numbers
    

class CustomDataset(Dataset):
    def __init__(self, data_zip_paths: list[tuple], index_dict: dict):
        self.data_zip_paths = data_zip_paths
        self.index_dict = index_dict
        
        # file 유효성 검사
        self.usable_zip_paths = []
        for path_pair in tqdm(data_zip_paths, desc="Validating ligand files..."):
            vox_path, lig_path = path_pair
            if check_usable_lig(lig_path):
                self.usable_zip_paths.append(path_pair)
        
        print(f"Dataset initialized: {len(self.usable_zip_paths)} valid files from {len(data_zip_paths)}")
            
    def __getitem__(self, index):
        sample = {}
        
        vox_path, lig_path = self.usable_zip_paths[index]
        
        lig_basename = os.path.basename(lig_path)
        if "pkl" in lig_basename:
            data_key = lig_basename.replace("_ligand.pkl", "")
        elif "mol2" in lig_basename:
            data_key = lig_basename.replace("_ligand.mol2", "")
        elif "sdf" in lig_basename:
            data_key = lig_basename.replace("_ligand.sdf", "")
        
        lig_mol = create_mol_from_file(lig_path, retry_without_sanitize=True)
                
        b_aff = None
        if self.index_dict:
            b_aff = self.index_dict.get(data_key, None)
        
        # protein 처리
        with open(vox_path, 'rb') as fp:
            voxels = pickle.load(fp)
            
        structure = voxels[..., :-1]
        pocket_label = voxels[..., -1:]
        
        sample['data_key'] = data_key
        sample["voxel"] = np.transpose(structure, (3, 0, 1, 2)) # channel-first expression
        sample["pocket_label"] = np.transpose(pocket_label, (3, 0, 1, 2)) # channel-first expression
        sample["ligand_data"] = encode_ligand_to_Data(lig_mol)
        sample["b_aff"] = b_aff

        return sample
    
    def __len__(self):
        return len(self.usable_zip_paths)
    
    
# class PreloadedDataset(Dataset):
#     """Under-developing."""
#     def __init__(self, data_zip_paths: list[tuple], index_dict: dict):
#         self.index_dict = index_dict
#         self.filtered_paths = []
#         self.preloaded_data = []
        
#         print(f"시작: 데이터셋 유효성 검사 및 메모리 로드 ({len(data_zip_paths)} 파일)")
        
#         # 유효성 검사 및 데이터 미리 로드
#         for path_pair in tqdm(data_zip_paths, desc="데이터 로딩 중"):
#             vox_path, lig_path = path_pair
            
#             # 리간드 유효성 검사
#             if not self._is_valid_ligand(lig_path):
#                 continue
                
#             # 데이터 로드
#             try:
#                 data_key = os.path.basename(os.path.dirname(lig_path))
                
#                 # Protein voxel 로드
#                 with open(vox_path, 'rb') as fp:
#                     voxels = pickle.load(fp)
                
#                 structure = voxels[..., :11]
#                 pocket_label = voxels[..., -1:]
                
#                 # Ligand 데이터 로드
#                 ligand_data = encode_ligand_to_Data(lig_path)
                
#                 # 결합 친화도 정보
#                 b_aff = None
#                 if self.index_dict:        
#                     b_aff = self.index_dict.get(data_key, None)
                
#                 # 전처리된 데이터 저장
#                 sample = {
#                     'data_key': data_key,
#                     "voxel": np.transpose(structure, (3, 0, 1, 2)),
#                     "pocket_label": np.transpose(pocket_label, (3, 0, 1, 2)),
#                     "ligand_data": ligand_data,
#                     "b_aff": b_aff
#                 }
                
#                 self.preloaded_data.append(sample)
#                 self.filtered_paths.append(path_pair)
                
#             except Exception as e:
#                 print(f"데이터 로드 실패: {vox_path}, {lig_path}, 오류: {e}")
        
#         print(f"완료: {len(self.preloaded_data)}/{len(data_zip_paths)} 데이터 로드됨")
        
#         # 메모리 사용량 대략 계산
#         sample_mem = sum(sys.getsizeof(x) for x in self.preloaded_data[0].values()) if self.preloaded_data else 0
#         total_mem = sample_mem * len(self.preloaded_data) / (1024**3)  # GB 단위
#         print(f"예상 메모리 사용량: 약 {total_mem:.2f} GB")
    
#     def _is_valid_ligand(self, lig_path):
#         """리간드 파일 유효성 검사"""
#         try:
#             # 분자 객체 생성
#             mol = create_mol_from_file(lig_path, retry_without_sanitize=True)
#             if mol is None:
#                 return False
            
#             # 분자 기본 검증
#             if mol.GetNumAtoms() == 0 or not mol.GetConformer().Is3D():
#                 return False
            
#             # 수소 제거 시도
#             mol_processed = None
#             rm_hydrogen_methods = [
#                 lambda m: Chem.RemoveHs(m),
#                 lambda m: Chem.RemoveHs(m, implicitOnly=True),
#                 lambda m: Chem.RemoveHs(m, implicitOnly=True, sanitize=False)
#             ]
            
#             for method in rm_hydrogen_methods:
#                 try:
#                     mol_processed = method(mol)
#                     break
#                 except:
#                     continue
            
#             return mol_processed is not None
            
#         except Exception:
#             return False
    
#     def __getitem__(self, index):
#         return self.preloaded_data[index]
    
#     def __len__(self):
#         return len(self.preloaded_data)