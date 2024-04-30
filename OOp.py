import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.data = ['Geography', 'Gender']
        self.label_encoder = LabelEncoder()
        self._preprocess_data()
        self._split_data()

    def _preprocess_data(self):
        self.df['CreditScore'] = self.df['CreditScore'].fillna(self.df['CreditScore'].median())
        for col in self.data:
            self.df[col] = self.label_encoder.fit_transform(self.df[col])
        self.df.drop(columns=['Surname', 'id', 'Unnamed: 0', 'CustomerId'], axis=1, inplace=True)

    def _split_data(self):
        X = self.df.drop('churn', axis=1)
        y = self.df['churn']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_random_forest(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        rf_pred = rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        return rf_model, rf_accuracy

    def train_xgboost(self):
        xgb_model = XGBClassifier()
        xgb_model.fit(self.X_train, self.y_train)
        xgb_pred = xgb_model.predict(self.X_test)
        xgb_accuracy = accuracy_score(self.y_test, xgb_pred)
        return xgb_model, xgb_accuracy

    def train_best_model(self):
        rf_model, rf_accuracy = self.train_random_forest()
        xgb_model, xgb_accuracy = self.train_xgboost()

        if rf_accuracy > xgb_accuracy:
            self.best_model = rf_model
            print("Model terbaik adalah Random Forest")
            self.best_accuracy = rf_accuracy
        else:
            self.best_model = xgb_model
            print("Model terbaik adalah XGBoost")
            self.best_accuracy = xgb_accuracy

    def save_best_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.best_model, f)

# Contoh penggunaan
trainer = ModelTrainer("data_C.csv")
trainer.train_best_model()
trainer.save_best_model("best_model.pkl")
