import os
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils import save_obj,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(project_root,"artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        X_train, y_train, X_test, y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
        
        models = {
            "Linear Regression" : LinearRegression()
        }

        model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        save_obj(
            file_path = self.model_trainer_config.trained_model_file_path,
            obj = best_model
        )