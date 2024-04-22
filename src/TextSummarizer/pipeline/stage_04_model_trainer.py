from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.model_trainer import ModelTrainer
from TextSummarizer.logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):         
        self.stage_name = "Model Trainer"

    def run(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        if model_trainer_config.active:
            logger.info(f">>>>>> stage {self.stage_name} started <<<<<<") 
            model_trainer = ModelTrainer(model_trainer_config)
            model_trainer.train()
            logger.info(f">>>>>> stage {self.stage_name} completed <<<<<<")
        else:
            logger.info(f"Previously completed {self.stage_name} - skip this stage")