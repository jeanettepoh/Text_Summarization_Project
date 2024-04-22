import os
import pandas as pd
from TextSummarizer.logging import logger
from TextSummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = None
            all_files = os.listdir(self.config.data_path)

            for file in all_files:
                if file not in self.config.all_required_files:
                    validation_status = False
                    with open(self.config.status_file, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.status_file, "w") as f:
                        f.write(f"Validation status: {validation_status}")

        except Exception as e:
            logger.error(e)
            raise e