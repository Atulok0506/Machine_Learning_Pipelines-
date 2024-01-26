from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, model, params):
    try:
        report = {}
        for model_name, model_instance in model.items():
            GS = GridSearchCV(model_instance, params[model_name], cv=5)
            GS.fit(X_train, y_train)

            model_instance.set_params(**GS.best_params_)
            model_instance.fit(X_train, y_train)

            y_pred = model_instance.predict(X_test)
            test_model_accuracy = accuracy_score(y_test, y_pred)
            report[model_name] = test_model_accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys)
