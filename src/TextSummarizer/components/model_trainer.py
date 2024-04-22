import os
import torch
from TextSummarizer.logging import logger
from TextSummarizer.entity import ModelTrainerConfig
from datasets import load_from_disk
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          TrainingArguments,
                          Trainer)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):

        # 1. Set up device, model, tokenizer, data collator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        # 2. Load dataset
        logger.info("Loading data...")
        dataset_pt = load_from_disk(self.config.data_path)

        # 3. Set up Training Arguments and Trainer
        trainer_args = TrainingArguments(
            output_dir = self.config.root_dir, 
            num_train_epochs = self.config.num_train_epochs, 
            warmup_steps = self.config.warmup_steps,
            per_device_train_batch_size = self.config.per_device_train_batch_size, 
            per_device_eval_batch_size = self.config.per_device_train_batch_size,
            weight_decay = self.config.weight_decay, 
            logging_steps = self.config.logging_steps,
            evaluation_strategy = self.config.evaluation_strategy, 
            eval_steps = self.config.eval_steps, 
            save_steps = self.config.save_steps,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps
        ) 

        trainer = Trainer(
            model = model,
            args = trainer_args,
            tokenizer = tokenizer,
            data_collator = seq2seq_data_collator,
            train_dataset = dataset_pt["train"],
            eval_dataset = dataset_pt["test"]
        )

        # 4. Commence training
        trainer.train()

        # 5. Save model and tokenizer
        logger.info("Saving model and tokenizer")
        model.save_pretrained(os.path.join(self.config.root_dir, "model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer")) 