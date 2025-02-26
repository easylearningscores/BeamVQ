import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
import logging
import os
import numpy as np
import h5py  #
from Beam_model import BeamVQ
from dataloader_api.dataloader_nse import load_navier_stokes_data



def augment_train_loader(model, train_loader, eval_loader, config):
    model.eval()
    saved_inputs = []
    saved_outputs = []
    
    with torch.no_grad():
        for inputs, _ in eval_loader:
            inputs = inputs.to(config.device)
            outputs, top_k_features = model(inputs)
            for top_k_feat in top_k_features:
                saved_inputs.append(inputs.cpu())
                saved_outputs.append(top_k_feat.cpu())
    
    if saved_inputs and saved_outputs:
        all_inputs = torch.cat(saved_inputs, dim=0)
        all_outputs = torch.cat(saved_outputs, dim=0)
        augmented_dataset = TensorDataset(all_inputs, all_outputs)
        augmented_loader = DataLoader(augmented_dataset, batch_size=config.batch_size, shuffle=True)
        
        combined_dataset = ConcatDataset([train_loader.dataset, augmented_dataset])
        combined_loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True)
        return combined_loader
    else:
        return train_loader


def evaluate_model(model, eval_loader, criterion, config):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(eval_loader, desc='Evaluating', leave=False):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

def train_model(model, train_loader, eval_loader, criterion, optimizer, config):
    best_loss = float('inf')
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', leave=True):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            optimizer.zero_grad()
            outputs, top_k_features = model(inputs)
            loss = criterion(outputs, targets)
            for top_k_feat in top_k_features:
                loss += criterion(top_k_feat, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        average_loss = total_loss / total_samples
        logging.info(f'Epoch {epoch + 1}, Train Loss: {average_loss:.7f}')

        eval_loss = evaluate_model(model, eval_loader, criterion, config)
        logging.info(f'Epoch {epoch + 1}, Eval Loss: {eval_loss:.7f}')

        if eval_loss < best_loss:
            best_loss = eval_loss
            logging.info(f'New best model found at epoch {epoch + 1} with loss {best_loss:.7f}. Saving model...')
            torch.save(model.state_dict(), f'{config.result_folder}/{config.backbone}_best_model_weights.pth')

        if (epoch + 1) % 100 == 0:
            logging.info(f"Epoch {epoch + 1}: augmenting training dataset.")
            train_loader = augment_train_loader(model, train_loader, eval_loader, config)
            logging.info(f"train dataset size: {len(train_loader)}")

    logging.info("Training complete.")

# lr schedule

def test_model(model, test_loader, criterion, topk, config):
    if not os.path.exists(config.result_folder):
        os.makedirs(config.result_folder)

    device = config.device
    model.to(device)
    model_path = f'{config.result_folder}/{config.backbone}_best_model_weights.pth'
    
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"No saved model found at {model_path}. Please check the path or ensure training has completed successfully.")
        return

    model.eval()
    test_loss = 0.0
    all_inputs = []
    all_targets = []
    all_preds = []
    all_top_k_features = [[] for _ in range(topk)] 

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, top_k_features = model(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

            for i, quantized_top_k in enumerate(top_k_features):
                if quantized_top_k.numel() > 0:  # Check if tensor is not empty
                    all_top_k_features[i].append(quantized_top_k.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.7f}')
    logging.info(f'Test Loss: {test_loss:.7f}')
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_top_k_features = [np.concatenate(top_k_feat, axis=0) if top_k_feat else np.array([]) for top_k_feat in all_top_k_features]

    result_file_path = f'{config.result_folder}/{config.backbone}_results.h5'
    with h5py.File(result_file_path, 'w') as f:
        f.create_dataset('inputs', data=all_inputs)
        f.create_dataset('targets', data=all_targets)
        f.create_dataset('preds', data=all_preds)

        for i, top_k_feat in enumerate(all_top_k_features):
            f.create_dataset(f'top_k_feat_{i}', data=top_k_feat)

    return test_loss
