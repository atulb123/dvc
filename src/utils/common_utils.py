import yaml
import json
import shutil
import logging
import os


def read_params(config_path: str = None) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def clean_dir_if_exists(dir_path: str):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def create_dir(dirs: list):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def save_local_df(df, df_path, header=False):
    if header:
        new_cols = [col.replace(" ", "_") for col in df.columns]
        df.to_csv(df_path, index=None, header=new_cols)
    else:
        df.to_csv(df_path, index=None)


def save_reports(file_path, report):
    with open(file_path, "w") as f:
        json.dump(report, f, indent=4)
