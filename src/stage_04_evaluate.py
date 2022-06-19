import shutil

import pandas as pd
import argparse
from src.utils.common_utils import read_params, create_dir, save_reports
from sklearn.model_selection import train_test_split
import logging
from sklearn.linear_model import ElasticNet
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate_metrics(actual, predicted):
    return mean_squared_error(actual, predicted), mean_absolute_error(actual, predicted), r2_score(actual, predicted)


def evaluate(config_path):
    config = read_params(config_path)
    artifacts = config["artifacts"]
    split_data = artifacts["split_data"]
    test_data_path = split_data["test_path"]
    model_path = artifacts["model_path"]
    base = config["base"]
    target_column = base["target_column"]
    score_file = artifacts["reports"]["scores"]
    test = pd.read_csv(test_data_path)
    test_x = test.drop(target_column, axis=1)
    test_y = test[target_column]
    lr = joblib.load(model_path)
    predicted_values = lr.predict(test_x)
    rmse, mae, r2 = evaluate_metrics(actual=test_y, predicted=predicted_values)
    scores = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    save_reports(score_file, scores)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    evaluate(parsed_args.config)
