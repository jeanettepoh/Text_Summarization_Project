from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.model_evaluation import ModelEvaluation
from TextSummarizer.logging import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        self.stage_name = "Model Evaluation"

    def run(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        if model_evaluation_config.active:
            logger.info(f">>>>>> stage {self.stage_name} started <<<<<<") 
            model_evaluation = ModelEvaluation(model_evaluation_config)
            model_evaluation.evaluate()
            logger.info(f">>>>>> stage {self.stage_name} completed <<<<<<")
        else:
            logger.info(f"Previously completed {self.stage_name} - skip this stage")