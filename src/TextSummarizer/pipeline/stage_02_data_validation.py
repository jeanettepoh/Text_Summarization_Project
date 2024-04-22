from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.data_validation import DataValidation
from TextSummarizer.logging import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        self.stage_name = "Data Validation"
        
    def run(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()

        if data_validation_config.active:
            logger.info(f">>>>>> stage {self.stage_name} started <<<<<<") 
            data_validation = DataValidation(data_validation_config)
            data_validation.validate_all_files_exist()
            logger.info(f">>>>>> stage {self.stage_name} completed <<<<<<")
        else:
            logger.info(f"Previously completed {self.stage_name} - skip this stage")
