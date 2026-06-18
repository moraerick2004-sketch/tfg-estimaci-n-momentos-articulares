"""
dataloader_camargo.py
=====================
Adapted version of Molinaro's TcnDataset that works with the converted
Camargo data. Simplified to work trial-by-trial (no batch zero-padding
needed for training).

This keeps the same interface but is more flexible with column names
and handles missing columns gracefully.
"""

import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CamargoTcnDataset(Dataset):
    """
    Dataset for loading converted Camargo data in Molinaro format.
    
    Expected directory structure:
        data_dir/
            AB01/
                trial_0000/
                    Exo.csv
                    Joint_Moments_Filt.csv
                trial_0001/
                    ...
            AB02/
                ...
    
    Parameters
    ----------
    data_dir : str
        Root directory of the converted data
    input_names : list of str
        Column names for input features
    label_names : list of str
        Column names for label/target features
    side : str
        'r' or 'l' (for Camargo, always 'r')
    participant_masses : dict
        Mapping of subject name to body mass (kg)
    device : torch.device
        Device for tensors
    """
    
    def __init__(self,
                 data_dir: str,
                 input_names: List[str],
                 label_names: List[str],
                 side: str = 'r',
                 participant_masses: Dict[str, float] = None,
                 device: torch.device = torch.device("cpu")):
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.participant_masses = participant_masses or {}
        self.device = device
        self.trial_paths = self._find_trials()
        
        if len(self.trial_paths) == 0:
            print(f"WARNING: No valid trials found in {data_dir}")
            print(f"  Looking for input columns: {input_names[:3]}...")
            print(f"  Looking for label columns: {label_names}")
    
    def __len__(self):
        return len(self.trial_paths)
    
    def __getitem__(self, idx):
        """
        Returns (input_tensor, label_tensor, sequence_length)
        
        Tensors are shaped (1, n_channels, sequence_length)
        """
        if isinstance(idx, (list, slice)):
            # Handle multi-index loading with zero padding
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
            else:
                indices = idx
            
            data = [self._load_single(i) for i in indices]
            data, seq_lengths = self._add_zero_padding(data)
            
            inputs = torch.cat([d[0] for d in data], dim=0)
            labels = torch.cat([d[1] for d in data], dim=0)
            
            return inputs, labels, seq_lengths
        else:
            inp, lab = self._load_single(idx)
            seq_length = inp.shape[-1]
            return inp, lab, [seq_length]
    
    def get_trial_names(self):
        return [os.path.basename(p) for p in self.trial_paths]
    
    def _find_trials(self) -> List[str]:
        """Find all valid trial directories."""
        trials = []
        
        if not os.path.exists(self.data_dir):
            print(f"ERROR: Data directory does not exist: {self.data_dir}")
            return trials
        
        # Look for subject directories
        for subject in sorted(os.listdir(self.data_dir)):
            subject_dir = os.path.join(self.data_dir, subject)
            if not os.path.isdir(subject_dir) or subject.startswith('.'):
                continue
            
            # Look for trial directories within each subject
            for trial in sorted(os.listdir(subject_dir)):
                trial_dir = os.path.join(subject_dir, trial)
                if not os.path.isdir(trial_dir):
                    continue
                
                # Check that both Exo.csv and Joint_Moments_Filt.csv exist
                exo_path = os.path.join(trial_dir, "Exo.csv")
                moments_path = os.path.join(trial_dir, "Joint_Moments_Filt.csv")
                
                if os.path.exists(exo_path) and os.path.exists(moments_path):
                    # Quick validation: check that required columns exist
                    try:
                        exo_cols = set(pd.read_csv(exo_path, nrows=0).columns)
                        mom_cols = set(pd.read_csv(moments_path, nrows=0).columns)
                        
                        # Check that at least some input and label columns exist
                        input_found = sum(1 for c in self.input_names if c in exo_cols)
                        label_found = sum(1 for c in self.label_names if c in mom_cols or c in exo_cols)
                        
                        if input_found > 0 and label_found > 0:
                            trials.append(trial_dir)
                        else:
                            if input_found == 0:
                                pass  # Silently skip - wrong stage data
                    except Exception:
                        continue
        
        return trials
    
    def _load_single(self, idx: int):
        """Load a single trial and return (input_tensor, label_tensor)."""
        trial_dir = self.trial_paths[idx]
        
        # Load input data
        exo_path = os.path.join(trial_dir, "Exo.csv")
        exo_df = pd.read_csv(exo_path)
        
        # Select only the columns we need (fill missing with NaN)
        input_data = pd.DataFrame()
        for col in self.input_names:
            if col in exo_df.columns:
                input_data[col] = exo_df[col].values
            else:
                print(f"WARNING: Column '{col}' not found in {exo_path}")
                input_data[col] = np.nan
        
        # Load label data (check both Exo.csv and Joint_Moments_Filt.csv)
        moments_path = os.path.join(trial_dir, "Joint_Moments_Filt.csv")
        moments_df = pd.read_csv(moments_path)
        
        label_data = pd.DataFrame()
        for col in self.label_names:
            if col in moments_df.columns:
                label_data[col] = moments_df[col].values
            elif col in exo_df.columns:
                label_data[col] = exo_df[col].values
            else:
                print(f"WARNING: Column '{col}' not found in either file")
                label_data[col] = np.nan
        
        # Ensure same length
        min_len = min(len(input_data), len(label_data))
        input_data = input_data.iloc[:min_len]
        label_data = label_data.iloc[:min_len]
        
        # Convert to tensors: (1, n_channels, sequence_length)
        input_tensor = torch.tensor(
            input_data.values, device=self.device, dtype=torch.float32
        ).transpose(0, 1).unsqueeze(0)
        
        label_tensor = torch.tensor(
            label_data.values, device=self.device, dtype=torch.float32
        ).transpose(0, 1).unsqueeze(0)
        
        return input_tensor, label_tensor
    
    def _add_zero_padding(self, data):
        """Add zero padding for batch loading."""
        seq_lengths = [d[0].shape[-1] for d in data]
        max_len = max(seq_lengths)
        
        padded = []
        for inp, lab in data:
            curr_len = inp.shape[-1]
            if curr_len < max_len:
                pad_len = max_len - curr_len
                inp = torch.cat([inp, torch.zeros(1, inp.shape[1], pad_len, device=self.device)], dim=2)
                lab = torch.cat([lab, torch.zeros(1, lab.shape[1], pad_len, device=self.device)], dim=2)
            padded.append((inp, lab))
        
        return padded, seq_lengths
