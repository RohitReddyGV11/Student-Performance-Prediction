import os
import dill
from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,"wb") as file_obj:
        dill.dump(obj,file_obj)


def evaluate_models(X_train,y_train,X_test,y_test,models):
    report = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        test_model_score = r2_score(y_test,y_test_pred)

        report[name] = test_model_score

    return report