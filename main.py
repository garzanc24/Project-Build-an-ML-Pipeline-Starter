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
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
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
            # Set the path for the artifact to clean
            input_artifact = "sample.csv"  # This would come from W&B or config
        
            # You may pass dynamic parameters from the config file, e.g., min_price, max_price
            output_artifact = "clean_sample.csv"
            output_type = "clean_data"
            output_description = "Cleaned dataset with outliers removed"
            min_price = config["etl"]["min_price"]
            max_price = config["etl"]["max_price"]
        
            # Call the basic_cleaning step and pass parameters to run.py
            _ = mlflow.run(
                f"{config['main']['components_repository']}/basic_cleaning",  # Make sure path to basic_cleaning is correct
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "input_artifact": input_artifact,
                    "output_artifact": output_artifact,
                    "output_type": output_type,
                    "output_description": output_description,
                    "min_price": min_price,
                    "max_price": max_price,
                },
            )


        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
