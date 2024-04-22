from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    active: bool
    root_dir: Path
    dataset_name: str
    dataset_split: str
    test_ratio: float


@dataclass(frozen=True)
class DataValidationConfig:
    active: bool
    root_dir: Path
    data_path: Path
    status_file: Path
    all_required_files: list


@dataclass(frozen=True)
class DataTransformationConfig:
    active: bool
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    input_max_length: int
    input_truncation: bool
    target_max_length: int
    target_truncation: bool


@dataclass(frozen=True)
class ModelTrainerConfig:
    active: bool
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    active: bool
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    batch_size: int
    input_max_length: int
    input_truncation: bool
    length_penalty: float
    num_beams: int
    target_max_length: int