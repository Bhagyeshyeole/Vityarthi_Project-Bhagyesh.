from src.loader import DataLoader
from src.cleaner import DataCleaner
from src.visualizer import DataVisualizer
from src.model import MLModel
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [iris|diabetes]")
        return
    
    dataset_choice = sys.argv[1]

    loader = DataLoader()
    cleaner = DataCleaner()
    visualizer = DataVisualizer()
    ai_model = MLModel()
    
    try:
        df = loader.load_dataset(dataset_choice)
        
        df_clean = cleaner.clean_data(df)
        
        visualizer.plot_data(df_clean)
        
        ai_model.train_predict(df_clean)
        
    except Exception as e:
        print(f" Error: {e}")

if __name__ == "__main__":
    main()