import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import logging
import os
import h5py
import numpy as np
from Beam_model import BeamVQ
from dataloader_api.dataloader import load_data
from model_training import train_model, test_model, evaluate_model

# ==========================================Config Class==========================================
# Configuration class to set up the model parameters and training environment
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 25
        self.learning_rate = 0.001
        self.num_epochs = 1
        self.backbone = 'cno'
        self.benchmark = 'nse'
        self.complete = False
        self.dataset_path = "dataloader_api/NavierStokes_V1e-5_N1200_T20.mat"
        self.result_folder = f'{self.backbone}_result/{self.benchmark}/{self.complete}'
        self.shape_in = (10, 1, 64, 64) 
        self.top_k = 10
        

config = Config()

# ==========================================Load Dataset==========================================

# train_loader, eval_loader, test_loader = load_data(path=config.dataset_path, batch_size=config.batch_size)

train_loader, eval_loader, test_loader = load_data(dataname=config.benchmark, 
                                                   batch_size = config.batch_size, 
                                                   val_batch_size=config.batch_size, 
                                                   data_root=config.dataset_path,
                                                   num_workers=8)

# ==========================================Load Model==========================================

model = BeamVQ(shape_in=config.shape_in, backbone_name=config.backbone, top_k=config.top_k, complete=config.complete)
model.to(config.device)

# ==========================================Configuration of the loss criterion and the optimizer for training==========================================
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) 

if not os.path.exists(config.result_folder):
    os.makedirs(config.result_folder)
logging.basicConfig(filename=f'{config.result_folder}/{config.backbone}{config.complete}_training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================training==========================================
train_model(model, train_loader, eval_loader, criterion, optimizer, config)

# ==========================================testing==========================================
test_model(model, test_loader, criterion, topk=config.top_k, config=config)
