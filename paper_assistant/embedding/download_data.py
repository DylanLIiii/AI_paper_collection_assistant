import logging
import os
import shutil
import kagglehub

logging.basicConfig(level=logging.INFO)

BASE_DATA_PATH = "/datadrive2/hengl/data"


def donwload_data():
    # check if using cache
    if check_data_exist():
        logging.info("Data already exists, skipping download")
        return

    dataset_path = os.path.join(BASE_DATA_PATH, "arxiv")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    # move the file under path to the BASE_DATA_PATH
    try:
        shutil.move(path, dataset_path)
    except Exception as e:
        logging.error(f"Error moving file to {dataset_path}: {e}")

    logging.info(f"Downloaded data to {dataset_path}")
    logging.info(f"Dataset Size: {os.path.getsize(dataset_path)}")


def check_data_exist():
    data_path = "/datadrive2/hengl/data/arxiv/212/arxiv-metadata-oai-snapshot.json"
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at {data_path}")
        return False
    return True


if __name__ == "__main__":
    donwload_data()
