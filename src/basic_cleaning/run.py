#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)
    
    # Download input artifact from W&B
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logging.info(f'The file "{artifact_local_path}" was successfully read.')
    
    # Filter the data: keep rows where price is between min_price and max_price
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    # Convert last_review to datetime format
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    # Save the cleaned data to a new CSV file
    clean_sample_file = "clean_sample.csv"
    df.to_csv(clean_sample_file, index=False)
    
    # Create a new artifact for the cleaned data and log it to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(clean_sample_file)
    run.log_artifact(artifact)
    logging.info(f'The file "{clean_sample_file}" was successfully uploaded to the W&B server.')
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Input artifact from the EDA step",
    )
    
    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Output artifact name for the cleaned data",
    )
    
    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
    )
    
    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
    )
    
    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to include in the cleaned dataset",
    )
    
    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to include in the cleaned dataset",
    )
    
    args = parser.parse_args()
    go(args)
