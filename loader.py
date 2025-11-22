from sklearn.datasets import load_iris, load_diabetes
import pandas as pd

class DataLoader:
    def load_dataset(self, dataset_name):

        print(f"[LOADER] Attempting to load {dataset_name}...")
        
        if dataset_name.lower() == 'iris':
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            # Rename for clarity
            df.columns = [c.replace(' (cm)', '').replace(' ', '_') for c in df.columns]
            df['dataset_type'] = 'classification'
            return df
            
        elif dataset_name.lower() == 'diabetes':
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['dataset_type'] = 'regression'
            return df
            
        else:
            raise ValueError("Dataset not supported. Choose 'iris' or 'diabetes'.")