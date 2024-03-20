import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from Dataset import RoadMarkingDataset

class Dataloader(pl.LightningDataModule):
    '''
    Dataloader class for the Mask2Former model


    '''
    def __init__(self, train_csv:str, val_csv:str, test_csv:str,batch_size:int , image_column:str, mask_column:str):
        self.__train_csv = train_csv
        self.__val_csv = val_csv
        self.__test_csv = test_csv
        self.__batch_size = batch_size
        self.__image_column = image_column
        self.__mask_column = mask_column
        # TO DO : Add the transforms (NORMALIZATION, RESIZE, TO TENSOR)
        self.__transform = None 
    
    def prepare_data(self):
        '''
        '''
    


    def setup(self, stage):
        '''
        Setup the dataloaders for the model.
        Args:
            stage: str, 'fit' or 'test'
        '''
        if (stage == 'fit'):
            self.__train_fit = RoadMarkingDataset(self.__train_csv, self.__image_column, self.__mask_column)
            self.__val_fit = RoadMarkingDataset(self.__val_csv, self.__image_column, self.__mask_column)
        
        if (stage == 'test'):
            self.__test_fit = RoadMarkingDataset(self.__test_csv, self.__image_column, self.__mask_column)
        

    def train_dataloader(self):
        '''
        Return the train dataloader
        '''
        return DataLoader(self.__train_fit, batch_size=self.__batch_size)
    
    def val_dataloader(self):
        '''
        Return the validation dataloader
        '''
        return Dataloader(self.__val_fit, batch_size=self.__batch_size)
    
    def test_dataloader(self):
        '''
        Return the test dataloader
        '''
        return Dataloader(self.__test_fit, batch_size=self.__batch_size)


