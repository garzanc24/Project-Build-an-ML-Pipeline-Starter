#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):
    """
    This function performs the data cleaning process:
    1. Downloads the raw dataset from W&B.
    2. Drops outliers based on price.
    3. Converts the 'last_review' column to datetime format.
    4. Filters rows based on longitude and latitude validity.
    5. Saves the cleaned data as 'clean_sample.csv' and logs it as a W&B artifact.
    """
    # Log the start of the basic cleaning step
    logger.info("Starting the basic cleaning step.")
    
    # Initialize W&B run
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact from W&B
    logger.info(f"Downloading artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(f"{args.input_artifact}:latest").file()
    df = pd.read_csv(artifact_local_path)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Drop outliers based on price using input parameters (min_price, max_price)
    logger.info(f"Filtering data for prices between {args.min_price} and {args.max_price}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert 'last_review' column to datetime
    logger.info("Converting 'last_review' column to datetime.")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Filter data for valid latitude and longitude
    logger.info("Filtering data based on valid latitude and longitude.")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned data to a CSV file
    logger.info("Saving cleaned data to 'clean_sample.csv'.")
    df.to_csv('clean_sample.csv', index=False)

    # Log the cleaned data as an artifact in W&B
    logger.info(f"Logging cleaned data as {args.output_artifact} to W&B.")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    logger.info(f"Basic cleaning step complete. Artifact {args.output_artifact} uploaded.")


# Completed Argument Section
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact (e.g., 'raw_data/sample.csv')",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact (e.g., 'clean_data')",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum accepted price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum accepted price",
        required=True
    )


    args = parser.parse_args()

    go(args)
