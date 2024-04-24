from TextSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        # 1. Define tokenizer, parameters, pipeline
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        gen_kwargs = {"length_penalty": self.config.length_penalty,
                      "num_beams": self.config.num_beams,
                      "max_length": self.config.target_max_length}
        
        pipe = pipeline("summarization", 
                        model = self.config.model_path, 
                        tokenizer =self.config.tokenizer_path)
        
        print("Dialogue:")
        print(text)

        # 2. Feed parameters and input text into pipe
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output