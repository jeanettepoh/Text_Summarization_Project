import torch
import pandas as pd
from tqdm import tqdm
from evaluate import load
from TextSummarizer.logging import logger
from TextSummarizer.entity import ModelEvaluationConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_from_disk

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def generate_batch_sized_chunks(self, list_of_elements):
        """
        Splits the dataset into smaller batches so that we can process simultaneously
        Yields successive batch-sized chunks from list_of_elements
        """
        for i in range(0, len(list_of_elements), self.config.batch_size):
            yield list_of_elements[i : i + self.config.batch_size]


    def calculate_metrics(self, dataset, metric, model, tokenizer):
        # 1. Generate batch sized chunks for text and summary
        text_batches = list(self.generate_batch_sized_chunks(dataset["text"]))
        summary_batches = list(self.generate_batch_sized_chunks(dataset["summary"]))

        # 2. Iterate over text and summary batches
        for text_batch, summary_batch in tqdm(
            zip(text_batches, summary_batches), total=len(text_batches)):

            # a. Tokenize inputs
            inputs = tokenizer(
                text_batch, 
                max_length = self.config.input_max_length,
                truncation = self.config.input_truncation,
                padding = "max_length",
                return_tensors = "pt"
            )

            # b. Generate summaries
            summaries = model.generate(
                input_ids = inputs["input_ids"].to(self.device),
                attention_mask = inputs["attention_mask"].to(self.device),
                max_length = self.config.target_max_length,
                num_beams = self.config.num_beams,
                length_penalty = self.config.length_penalty
            )

            # c. Decode summaries
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
                                 for s in summaries]
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            # d. Add prediction batch and summary batch to metric object
            metric.add_batch(predictions=decoded_summaries, references=summary_batch)

        # 3. Compute and return the ROUGE scores
        score = metric.compute()
        return score
    

    def evaluate(self):
        # 1. Load saved tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(self.device)

        # 2, Load dataset
        dataset_pt = load_from_disk(self.config.data_path)

        # 3. Set up rouge metric object
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = load("rouge")

        # 4. Evaluate on dataset using model, tokenizer, rouge_metric
        score = self.calculate_metrics(dataset_pt["test"][0:10], rouge_metric, model, tokenizer)

        # 5. Get ans save individual rouge metrics
        rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
        df = pd.DataFrame(rouge_dict, index = ['t5-small'] )
        df.to_csv(self.config.metric_file_name, index=False)