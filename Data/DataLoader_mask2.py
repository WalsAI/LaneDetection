from pyparsing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
import pandas as pd
import numpy as np
from Dataset import RoadMarkingDataset

class RoadMarkingDataloader(pl.LightningDataModule):
    '''
    Dataloader class for the Mask2Former model

    '''
    def __init__(self, train_csv:str, val_csv:str, test_csv:str,batch_size:int , image_column:str, mask_column:str):
        '''
        Constructor for the class
        '''
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.image_column = image_column
        self.mask_column = mask_column
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        # TO DO : Add the transforms (NORMALIZATION, RESIZE, TO TENSOR)
        self.__transform = None 

    def setup(self, stage: str):
        '''
        Setup the dataset for the dataloader
        '''
        if stage == 'fit':
            self.train_ds = RoadMarkingDataset(self.train_csv, self.image_column, self.mask_column)
        if stage == 'val':
            self.val_ds = RoadMarkingDataset(self.val_csv, self.image_column, self.mask_column)
        if stage == 'test':
            self.test_ds = RoadMarkingDataset(self.test_csv, self.image_column, self.mask_column)
    

    def train_dataloader(self):
        '''
        Train dataloader
        '''
        return DataLoader(self.train_ds, batch_size=self.batch_size)
    
    def val_dataloader(self):
        '''
        Validation dataloader
        
        '''
        return DataLoader(self.val_ds, batch_size=self.batch_size)
    
    def test_dataloader(self):
        '''
        Test dataloader
        '''
        return DataLoader(self.test_ds, batch_size=self.batch_size)


