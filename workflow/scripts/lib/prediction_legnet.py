

import torch 

import lightning.pytorch as pl
import numpy as np 

from lib.human_legnet.trainer import LitModel
from lib.human_legnet.training_config import TrainingConfig
from torch.utils.data import DataLoader, Dataset


class TrainSeqDatasetProb(Dataset):
    """np.array dataset"""
    
    def __init__(self, 
                 ds: np.ndarray,
                 reverse: bool,
                 use_reverse_channel: bool,  
                 seqsize=230):
      
        self.ds = torch.from_numpy(ds)
        # n x 4 x length
        assert ds.shape[2] == seqsize
        self.reverse = reverse
        self.use_reverse_channel = use_reverse_channel
        self.seqsize = seqsize 
    
    def __getitem__(self, i):
        #n x 4 x length
        seq = self.ds[i, :, :]
        
        if self.reverse:
            rev = 1.0
            seq = torch.flip(seq, (-1, -2))
        else:
            rev = 0.0
            
        to_concat = [seq]
        
        # add reverse augmentation channel
        if self.use_reverse_channel:
            rev = torch.full( (1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev)
            
        # create final tensor
        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq
        
        return X, -1
    
    def __len__(self):
        return len(self.ds)

def oh_predict(trainer: pl.Trainer,
               model: pl.LightningModule, 
               cfg: TrainingConfig,
               data: np.ndarray) -> np.ndarray:
    
    forw_ds = TrainSeqDatasetProb(data, 
                                  reverse=False,
                                  use_reverse_channel=cfg.use_reverse_channel)
    forw_dl = DataLoader(forw_ds, 
                         batch_size=cfg.valid_batch_size,
                         num_workers=cfg.num_workers,
                         shuffle=False)
    

    
    forw_y_preds =  trainer.predict(model,
                               dataloaders=forw_dl)
    forw_y_preds = torch.concat(forw_y_preds).cpu().numpy() #type: ignore
    
    if cfg.reverse_augment:
        rev_ds = TrainSeqDatasetProb(data, 
                                  reverse=True,
                                  use_reverse_channel=cfg.use_reverse_channel)
        rev_dl = DataLoader(rev_ds, 
                         batch_size=cfg.valid_batch_size,
                         num_workers=cfg.num_workers,
                         shuffle=False)
        
          
        rev_y_preds =  trainer.predict(model,
                                dataloaders=rev_dl)
        rev_y_preds = torch.concat(rev_y_preds).cpu().numpy() #type: ignore
        y_preds = (forw_y_preds + rev_y_preds) / 2
    else:
        y_preds = forw_y_preds
    return y_preds


def predict(model_file: str, weights_file: str, data: np.ndarray) -> np.ndarray:

    train_cfg = TrainingConfig.from_json(model_file)

    torch.set_float32_matmul_precision('medium') # type: ignore 

    model = LitModel.load_from_checkpoint(weights_file, 
                                        tr_cfg=train_cfg)

    

    data = data.astype(np.float16) # #n x length x 4 
    data = np.swapaxes(data, 1,2 ) #n x 4 x length
    

    trainer = pl.Trainer(accelerator='gpu',
                        devices=[0], 
                        precision='16-mixed')
    
    preds = oh_predict(trainer, model, train_cfg, data)
    preds = np.array(preds, dtype=np.float16)
    return(preds)
