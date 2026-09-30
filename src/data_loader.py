import pandas as pd
import os
import gc

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_source(self, filename, cols=None, nrows=None):
        """Efficiently load a TSV source file with optimized dtypes."""
        path = os.path.join(self.data_dir, filename)
        
        # Define types to save memory
        dtype_map = {
            'entity_id': 'string',
            'business_name': 'string',
            'business_address': 'string',
            'country': 'category'
        }
        
        df = pd.read_csv(
            path, 
            sep='\t', 
            usecols=cols, 
            dtype=dtype_map,
            nrows=nrows,
            engine='c'
        )
        return df

    def get_ground_truth(self):
        """Loads train_ground_truth and parses comma-separated matched IDs."""
        path = os.path.join(self.data_dir, 'train_ground_truth.tsv')
        df = pd.read_csv(path, sep='\t')
        # Handle cases where matched_entity_ids is NaN (no matches)
        df['matched_entity_ids'] = df['matched_entity_ids'].fillna('')
        df['matched_list'] = df['matched_entity_ids'].apply(
            lambda x: x.split(',') if x else []
        )
        return df
