from TextSummarizer.logging import logger
from TextSummarizer.entity import DataTransformationConfig
from transformers import AutoTokenizer
from datasets import load_from_disk

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    def convert_examples_to_features(self, example_batch):

        input_encodings = self.tokenizer(
            example_batch["text"],
            truncation = self.config.input_truncation, 
            max_length = self.config.input_max_length
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'], 
                truncation = self.config.target_truncation, 
                max_length = self.config.target_max_length
            )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }


    def load_and_convert(self):

        dataset = load_from_disk(self.config.data_path)
        logger.info(f"Dataset info: {dataset}") 
        dataset_pt = dataset.map(self.convert_examples_to_features, batched=True)
        dataset_pt.save_to_disk(self.config.root_dir)
        logger.info("Saved tokenized dataset to disk")