from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

class MLModel:
    def train_predict(self, df):
        print("[MODEL] Initializing training process...")
        X = df.drop(columns=['target', 'dataset_type'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dataset_type = df['dataset_type'].iloc[0]
        
        if dataset_type == 'classification':
            print("[MODEL] Task: Classification (KNN)")
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f" Model Accuracy: {acc:.2f}")
            
        elif dataset_type == 'regression':
            print("[MODEL] Task: Regression (Linear)")
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            print(f" Model Mean Squared Error: {mse:.2f}")