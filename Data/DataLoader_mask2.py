# This file contains the dataloader class for the Mask2Former model
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import EVAL_DATALOADERS,TRAIN_DATALOADERS
from Dataset import RoadMarkingDataset

class RoadMarkingDataloader(pl.LightningDataModule):
    '''
    Dataloader class for the Mask2Former model

    '''
    def __init__(self, train_csv:str, val_csv:str, test_csv:str,batch_size:int , image_column:str, mask_column:str):
        '''
        Constructor for the class
        Parameters:
        train_csv : str : Path to the training csv
        val_csv : str : Path to the validation csv
        test_csv : str : Path to the test csv
        batch_size : int : Batch size for the dataloader
        image_column : str : Column name for the image
        mask_column : str : Column name for the mask
        '''
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.image_column = image_column
        self.mask_column = mask_column
        self.stage_ds = {'train': None, 'val': None, 'test': None}
        # TO BE ADDED : Add the transforms (NORMALIZATION, RESIZE, TO TENSOR)
        self.__transform = None 

    def setup(self, stage: str):
        '''
        Setup the dataset for the dataloader
        Parameters:
        stage : str : Stage for which the dataset is to be created
        '''
        if stage == 'fit':
            self.stage_ds['train'] = RoadMarkingDataset(self.train_csv, self.image_column, self.mask_column)
            self.stage_ds['val'] = RoadMarkingDataset(self.val_csv, self.image_column, self.mask_column)
        if stage == 'test':
            self.stage_ds['test'] = RoadMarkingDataset(self.test_csv, self.image_column, self.mask_column)
    

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        '''
        Train dataloader
        '''
        return DataLoader(self.stage_ds['train'], batch_size=self.batch_size)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        '''
        Validation dataloader
        '''
        return DataLoader(self.stage_ds['val'], batch_size=self.batch_size)
    
    def test_dataloader(self):
        '''
        Test dataloader
        '''
        return DataLoader(self.stage_ds['test'], batch_size=self.batch_size)


