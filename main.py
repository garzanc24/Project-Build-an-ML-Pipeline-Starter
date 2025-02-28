import os
import hydra
from omegaconf import DictConfig
import mlflow

@hydra.main(config_path=".", config_name="config")
def go(config: DictConfig):
    # Split the steps defined in the config
    active_steps = config["main"]["steps"].split(",")

    # -----------------------------
    # Step: Download Data
    # -----------------------------
    if "download" in active_steps:
        _ = mlflow.run(
            f"{config['main']['components_repository']}/get_data",
            "main",
            version="main",
            env_manager="conda",
            parameters={
                "sample": config["etl"]["sample"],
                "artifact_name": "sample.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw file as downloaded"
            },
        )

    # -----------------------------
    # Step: Exploratory Data Analysis (EDA)
    # -----------------------------
    if "eda" in active_steps:
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "eda"),
            "main",
            env_manager="conda",
            parameters={},
        )

    # -----------------------------
    # Step: Basic Cleaning
    # -----------------------------
    if "basic_cleaning" in active_steps:
       _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
            "main",
            env_manager="conda",
            parameters={
                "input_artifact": "raw_data/sample.csv:latest"
                "output_artifact": "clean_sample.csv",
                "output_type": "dataset",
                "output_description": "Cleaned Airbnb dataset",
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"]
            },
        )


    # -----------------------------
    # Step: Data Check
    # -----------------------------
    if "data_check" in active_steps:
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
            "main",
            env_manager="conda",
            parameters={
                "csv": "clean_sample.csv:latest",
                "ref": "clean_sample.csv:reference",
                "kl_threshold": config["data_check"]["kl_threshold"],
                "min_price": config["etl"]["min_price"],
                "max_price": config["etl"]["max_price"]
            },
        )

    # -----------------------------
    # Step: Data Split
    # -----------------------------
    if "data_split" in active_steps:
        _ = mlflow.run(
            f"{config['main']['components_repository']}/train_val_test_split",
            "main",
            version="main",
            env_manager="conda",
            parameters={
                "input": "clean_sample.csv:latest",
                "test_size": "0.10",
                "random_seed": "42",
                "stratify_by": "none"
            },
        )

    # -----------------------------
    # Step: Train Random Forest
    # -----------------------------
    if "train_random_forest" in active_steps:
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
            "main",
            env_manager="conda",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "val_size": "0.10",
                "random_seed": "42",
                "stratify_by": "neighbourhood_group",
                "rf_config": config["modeling"]["random_forest"],
                "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                "output_artifact": "random_forest_export"
            },
        )

    # -----------------------------
    # Step: Test Regression Model
    # -----------------------------
    if "test_regression_model" in active_steps:
        _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "test_regression_model"),
            "main",
            env_manager="conda",
            parameters={
                "input_artifact": "random_forest_export:latest",
                "mlflow_model": "random_forest_model",
                "output_artifact": "regression_model_report"
            },
        )

if __name__ == "__main__":
    go()
