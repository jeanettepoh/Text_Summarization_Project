from TextSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from TextSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from TextSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from TextSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from TextSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from TextSummarizer.logging import logger


try:
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.run()
except Exception as e:
    logger.exception(e)
    raise e


try:
    data_validation = DataValidationTrainingPipeline()
    data_validation.run()
except Exception as e:
    logger.exception(e)
    raise e


try:
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.run()
except Exception as e:
    logger.exception(e)
    raise e


try:
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.run()
except Exception as e:
    logger.exception(e)
    raise e


try:
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.run()
except Exception as e:
    logger.exception(e)
    raise e