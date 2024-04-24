from TextSummarizer.constants import *
from TextSummarizer.utils.common import read_yaml, create_directories
from TextSummarizer.entity import (DataIngestionConfig,
                                   DataValidationConfig,
                                   DataTransformationConfig,
                                   ModelTrainerConfig,
                                   ModelEvaluationConfig) 

class ConfigurationManager:
    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
        params_file_path=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            active = config.active,
            root_dir = config.root_dir,
            dataset_name = config.dataset_name,
            dataset_split= config.dataset_split,
            test_ratio = config.test_ratio
        )

        return data_ingestion_config

    
    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            active = config.active,
            root_dir = config.root_dir,
            data_path = config.data_path,
            status_file = config.status_file,
            all_required_files = config.all_required_files
        )

        return data_validation_config


    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation
        params = self.params.Tokenizer

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            active = config.active,
            root_dir = config.root_dir,
            data_path = config.data_path,
            tokenizer_name = config.tokenizer_name,
            input_max_length = params.input_max_length,
            input_truncation = params.input_truncation,
            target_max_length = params.target_max_length,
            target_truncation = params.target_truncation
        )

        return data_transformation_config


    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            active = config.active,
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_ckpt = config.model_ckpt,
            num_train_epochs = params.num_train_epochs,
            warmup_steps = params.warmup_steps,
            per_device_train_batch_size = params.per_device_train_batch_size,
            weight_decay = params.weight_decay,
            logging_steps = params.logging_steps,
            evaluation_strategy = params.evaluation_strategy,
            eval_steps = params.eval_steps,
            save_steps = params.save_steps,
            gradient_accumulation_steps = params.gradient_accumulation_steps
        )

        return model_trainer_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation
        params = self.params.ModelEvaluation

        create_directories([config.root_dir])

        model_evalution_config = ModelEvaluationConfig(
            active = config.active,
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            metric_file_name = config.metric_file_name,
            batch_size = params.batch_size,
            input_max_length = params.input_max_length,
            input_truncation = params.input_truncation,
            length_penalty = params.length_penalty,
            num_beams = params.num_beams,
            target_max_length = params.target_max_length
        )

        return model_evalution_config