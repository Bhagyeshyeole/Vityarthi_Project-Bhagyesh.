import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def plot_data(self, df):
        print("[VISUALIZER] Generating charts...")
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df['target'], kde=True, color='blue')
        plt.title('Target Distribution')
        
        plt.subplot(1, 2, 2)

        numeric_df = df.drop(columns=['dataset_type'])
        sns.heatmap(numeric_df.corr(), annot=False, cmap='viridis')
        plt.title('Feature Correlation')
        
        plt.tight_layout()
        plt.show()