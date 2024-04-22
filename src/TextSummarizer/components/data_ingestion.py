from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from TextSummarizer.logging import logger
from TextSummarizer.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
 

    def load_and_save_data(self):

        logger.info(f"Loading dataset from huggingface")

        dataset = load_dataset(
            self.config.dataset_name, 
            split = self.config.dataset_split
        )

        split_dataset = self.split_data(dataset)
  
        logger.info(f"Dataset info: {split_dataset}") 

        # Save dataset to disk
        split_dataset.save_to_disk(self.config.root_dir)

        logger.info("Split and saved dataset in huggingface format")


    def split_data(self, dataset):

        split_dataset = dataset.train_test_split(test_size=self.config.test_ratio)
        train_data = split_dataset["train"]
        remaining_data = split_dataset["test"]

        split_remaining = remaining_data.train_test_split(test_size=0.5)
        validation_data = split_remaining["train"]
        test_data = split_remaining["test"]

        split_dataset = DatasetDict({
            "train": train_data,
            "validation": validation_data,
            "test": test_data
        })

        return split_dataset