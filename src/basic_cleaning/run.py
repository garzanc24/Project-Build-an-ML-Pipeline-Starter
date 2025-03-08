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

def go(args):
    run = wandb.init(
        job_type="basic_cleaning",
        project="nyc_airbnb", 
        group="cleaning", 
        save_code=True
    )
    run.config.update(args)

    logger.info(f"Downloading artifact: {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    
    logger.info("Dropping outliers based on price range")
    df = df[df['price'].between(args.min_price, args.max_price)].copy()
    
    logger.info("Converting last_review to datetime format")
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    
    logger.info("Filtering locations within NYC boundaries")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()



    
    output_path = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {output_path}")
    df.to_csv(output_path, index=False)
    
    logger.info("Logging cleaned data as artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)
    
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data cleaning for NYC Airbnb dataset")
    
    parser.add_argument("--input_artifact", type=str, help="Path to input artifact", required=True)
    parser.add_argument("--output_artifact", type=str, help="Name for output artifact", required=True)
    parser.add_argument("--output_type", type=str, help="Type of output artifact", required=True)
    parser.add_argument("--output_description", type=str, help="Description of output artifact", required=True)
    parser.add_argument("--min_price", type=float, help="Minimum accepted price", required=True)
    parser.add_argument("--max_price", type=float, help="Maximum accepted price", required=True)
    
    args = parser.parse_args()
    go(args)
