from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.data_transformation import DataTransformation
from TextSummarizer.logging import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        self.stage_name = "Data Transformation"

    def run(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()

        if data_transformation_config.active:
            logger.info(f">>>>>> stage {self.stage_name} started <<<<<<") 
            data_transformation = DataTransformation(data_transformation_config)
            data_transformation.load_and_convert()
            logger.info(f">>>>>> stage {self.stage_name} completed <<<<<<")
        else:
            logger.info(f"Previously completed {self.stage_name} - skip this stage")