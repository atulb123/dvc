import shutil

import pandas as pd
import argparse
from src.utils.common_utils import read_params, create_dir, save_reports
from sklearn.model_selection import train_test_split
import logging
from sklearn.linear_model import ElasticNet
import joblib


def train(config_path):
    config = read_params(config_path)
    artifacts = config["artifacts"]

    split_data = artifacts["split_data"]
    processed_data_dir = split_data["processed_data_dir"]
    train_data_path = split_data["train_path"]
    test_data_path = split_data["test_path"]

    base = config["base"]
    random_seed = base["random_state"]
    target_column = base["target_column"]

    reports = artifacts["reports"]
    reports_dir = reports["reports_dir"]
    params = reports["params"]
    elastic_net_parameters = config["estimators"]["ElasticNet"]["params"]
    alpha = elastic_net_parameters["alpha"]
    l1_ratio = elastic_net_parameters["l1_ratio"]

    train = pd.read_csv(train_data_path)
    train_y = train[target_column]
    train_x = train.drop(target_column, axis=1)
    lr_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_seed)
    lr_model.fit(train_x, train_y)
    model_dir = artifacts["model_dir"]
    model_path = artifacts["model_path"]
    create_dir([model_dir, reports_dir])
    params_value = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    save_reports(params, params_value)
    joblib.dump(lr_model, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train(parsed_args.config)
