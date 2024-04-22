from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.data_ingestion import DataIngestion
from TextSummarizer.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        self.stage_name = "Data Ingestion"
    
    def run(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        if data_ingestion_config.active:
            logger.info(f">>>>>> stage {self.stage_name} started <<<<<<") 
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion.load_and_save_data()
            logger.info(f">>>>>> stage {self.stage_name} completed <<<<<<")
        else:
            logger.info(f"Previously completed {self.stage_name} - skip this stage")