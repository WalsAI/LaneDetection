from typing import Union, List
import numpy as np

class PreprocessingBoundingBoxes:
    """
    Preprocessing bounding boxes from dataset in order to generate masks with Segment Anything Model (SAM) as states in `https://arxiv.org/pdf/2304.02643.pdf`.
    """
    def __init__(
        self,
        path_to_dataset: str,
        path_to_output_csv: str,
    ) -> None:
    """
    Initialization for preprocessing the bounding boxes in order
    to match Segment Anything Model (SAM) as states in LabelAnything repository `https://github.com/WalsAI/LabelAnything`.
    
    Args:
        path_to_dataset: Path to directory for labels (train/val/test).
        path_to_output_csv: Path to output csv file in format suggested in LabelAnything `https://github.com/WalsAI/LabelAnything`
    """
        self.path_to_dataset = path_to_dataset
        self.path_to_output_csv = path_to_output_csv

    def _preprocess_file(self, file_path: str) -> Union[List[str], List[np.ndarray], List[int]]:
        """
            Preprocess ground truth file and then add it to csv file.

            Args:
                file_path: Path to file for preprocessing.
        """
