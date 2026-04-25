"""
Module to download the Anime Face Dataset from Kaggle.
"""
import os
import kagglehub
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_dataset() -> str:
    """
    Downloads the dataset from Kaggle using kagglehub.
    
    Returns:
        str: The path to the downloaded dataset.
    """
    logger.info("Starting dataset download...")
    try:
        path = kagglehub.dataset_download("splcher/animefacedataset")
        logger.info(f"Dataset successfully downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

if __name__ == "__main__":
    download_dataset()
