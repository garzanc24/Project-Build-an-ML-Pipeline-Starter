import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model"
]

@hydra.main(config_path=".", config_name='config')
def go(config: DictConfig):
    # Setup the wandb experiment
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Determine steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Use a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        if "download" in active_steps:
            # Download file using the remote repository
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            # Define the root path
            root_path = hydra.utils.get_original_cwd()
            _ = mlflow.run(
                os.path.join(root_path, "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data_with_outliers_and_null_values_removed",  # Use underscores instead of spaces
                    "min_price": float(config["etl"]["min_price"]),
                    "max_price": float(config["etl"]["max_price"])
                },
            )

        # Placeholder for other steps
        if "data_check" in active_steps:
            # Define the root path for execution
            root_path = hydra.utils.get_original_cwd()
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": float(config["etl"]["min_price"]),
                    "max_price": float(config["etl"]["max_price"]),
                },
            )


        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": None if config["modeling"]["stratify_by"] == "none" else config["modeling"]["stratify_by"],

                },
            )


        if "train_random_forest" in active_steps:
            root_path = hydra.utils.get_original_cwd()  
        
            # Serialize Random Forest configuration from config.yaml
            rf_config = os.path.join(root_path, "rf_config.json")
            with open(rf_config, "w") as fp:
                json.dump(dict(config["modeling"]["random_forest"]), fp)
        
            _ = mlflow.run(
                os.path.join(root_path, "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "output_artifact": "random_forest_export",
                    "rf_config": rf_config,
                    "random_seed": config["modeling"]["random_seed"],
                    "val_size": config["modeling"]["val_size"],  
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],  
                },
            )





        if "test_regression_model" in active_steps:
            pass

if __name__ == "__main__":
    go()