# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 15:13:00 2024

@author: rriahisamani
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchvision.transforms import Resize
import torch.nn as nn
import numpy as np
import time
import json
import wandb
import pickle
import os
import copy
import random
import matplotlib.cm as cm
import os
from utils.cwt import CWT
from typing import Optional
import torch




filename = os.path.splitext(os.path.basename(__file__))[0]

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU-based operations
    np.random.seed(seed)
    
set_seeds(0)
def add_gaussian_noise(signal, nsr=0.10):
    set_seeds(0)
    signal_energy = np.mean(signal**2)
    noise_energy = nsr * signal_energy
    noise_std_dev = np.sqrt(noise_energy)
    noise = np.random.normal(0, noise_std_dev, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

class PreprocessedDataset(Dataset): 
    def __init__(self, data_dir, transform=None):
        self.data_files = sorted(os.listdir(data_dir))
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        sample = torch.load(file_path)  # dict {'data': X, 'label': y}
        data, label = sample['data'], sample['label']
        if self.transform:
            data = self.transform(data)
        data = data.float()   # convert to float32
        label = label.float()
        return data, label


class VibrationAugment:
    def __init__(self,
                 noise_std=0.05,
                 scale_range=(0.9, 1.1),
                 jitter_prob=0.5,
                 crop_prob=0.5,
                 crop_frac=0.9,
                 freq_mask_prob=0.5,
                 freq_mask_size=4):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.jitter_prob = jitter_prob
        self.crop_prob = crop_prob
        self.crop_frac = crop_frac
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_size = freq_mask_size

    def __call__(self, x):
        """
        x: Tensor of shape [1, L] or [C, L] (vibration signal)
        returns augmented tensor with the same shape
        """

        if random.random() < 0.5:
            x = x + self.noise_std * torch.randn_like(x)


        if random.random() < 0.5:
            factor = torch.empty(1).uniform_(*self.scale_range).item()
            x = x * factor

        if random.random() < self.jitter_prob:
            shift = random.randint(-5, 5)  # shift up to ±5 samples
            x = torch.roll(x, shifts=shift, dims=-1)

        if random.random() < self.crop_prob:
            L = x.size(-1)
            new_L = int(self.crop_frac * L)
            start = random.randint(0, L - new_L)
            x = x[..., start:start+new_L]
            # pad back to original length
            x = F.pad(x, (0, L - new_L))

        if random.random() < self.freq_mask_prob:
            L = x.size(-1)
            mask_size = min(self.freq_mask_size, L)
            start = random.randint(0, L - mask_size)
            x[..., start:start+mask_size] = 0

        return x





# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, input_sequences1, input_sequences2, labels, augment=None):
        self.input_sequences_1 = input_sequences1  # List of tensors with varying sizes
        self.input_sequences_2 = input_sequences2
        self.labels = labels                    # Tensor of labels
        self.augment = VibrationAugment(noise_std=0.05, scale_range=(0.9, 1.1), jitter_prob=0.5,
                 crop_prob=0.5,
                 crop_frac=0.9,
                 freq_mask_prob=0.6,
                 freq_mask_size=5)

    def __len__(self):
        return len(self.input_sequences_1)

    def __getitem__(self, idx):

        x1 = torch.tensor(self.input_sequences_1[idx], dtype=torch.float32)
        x2 = torch.tensor(self.input_sequences_2[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        # if self.augment is not None:
            # x1 = self.augment(x1)
            # x2 = self.augment(x2)

        return x1, x2, y
        

def collate_fn(batch):
    inputs, inputs_s, labels = zip(*batch)


    lengths = torch.tensor([seq.size(1) for seq in inputs])

    sorted_indices = torch.argsort(lengths, descending=True)
    inputs = [inputs[i] for i in sorted_indices]
    
    lengths_s = torch.tensor([seq.size(1) for seq in inputs_s])
    
    inputs_s = [inputs_s[i] for i in sorted_indices]

    inputs = torch.stack(inputs)


    transform = Resize((224,224))

    inputs= transform(inputs)

    labels = torch.stack([labels[i] for i in sorted_indices])

    max_len = lengths.max().item()
    max_len_s = lengths_s.max().item()

    padded_inputs_s = torch.stack([torch.nn.functional.pad(seq, (0, max_len_s - seq.shape[1]))
                                 for seq in inputs_s])

    lens = lengths[sorted_indices]
    return inputs, padded_inputs_s, labels, lens


def aba_segmentation(dataset_list):
    input_arrays = []
    input_sppeds = []
    for array1, _, _ in dataset_list:
        array1_aba = torch.from_numpy(array1[0])
        array1_spped = torch.from_numpy(array1[1])
        input_arrays.append(array1_aba.unsqueeze(0))
        input_sppeds.append(array1_spped.unsqueeze(0))
        
    return input_arrays, input_sppeds

def aba_segmentation_t(dataset_list):
    input_arrays = []
    input_sppeds = []
    for array1, _, _ in dataset_list:

        array_aba = np.array(array1[0], dtype=np.float32)
        array_aba = torch.from_numpy(array_aba)
        array_speed = np.array(array1[1], dtype=np.float32)
        array_speed = torch.from_numpy(array_speed)

        input_arrays.append(array_aba.unsqueeze(0))
        input_sppeds.append(array_speed.unsqueeze(0))
        
    return input_arrays, input_sppeds



def Data_loaders(bs, force=False):


    train_dataset  = PreprocessedDataset("/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Benchmarks/utils/CWT_out2/train")
    val_dataset = PreprocessedDataset("/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Benchmarks/utils/CWT_out2/val")
    test_dataset = PreprocessedDataset("/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Benchmarks/utils/CWT_out2/test")


    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader




#model defination 



class _ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class InceptionV1Module(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionV1Module, self).__init__()

        # 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # 1x1 -> 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 1x1 -> 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # 3x3 pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x),
                   self.branch3(x), self.branch4(x)]
        return torch.cat(outputs, 1)


class GoogLeNetV1(nn.Module):
    def __init__(self, num_classes=30, aux_logits=True, dropout=0.4):
        super(GoogLeNetV1, self).__init__()
        self.aux_logits = aux_logits


        self.transform=Resize((224,224))
        # Initial layers (stem)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception modules (as in paper)
        self.inception3a = InceptionV1Module(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionV1Module(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = InceptionV1Module(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionV1Module(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionV1Module(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionV1Module(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionV1Module(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = InceptionV1Module(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionV1Module(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

        self.aux1 = self.aux2 = None

    def forward(self, x):



        x=self.transform(x)


        # Stem
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)

        # Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        x = self.fc(x)


        x = x.view(-1, 30, 1)

        return x



    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")    

epsilon = 1e-9 


# Function to count the total parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_mape(preds, actuals):
    absolute_percentage_error = torch.abs((preds - actuals) / (actuals + epsilon)) * 100
    return absolute_percentage_error.mean().item()

def calculate_r2(preds, actuals):
    total_sum_of_squares = torch.sum((actuals - actuals.mean(dim=0)) ** 2, dim=0)
    residual_sum_of_squares = torch.sum((actuals - preds) ** 2, dim=0)
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2.mean().item()


def MSE(predictions, actuals):
    mse=torch.mean((predictions-actuals)**2, dim=0)
    return mse




def train_and_val(model, train_dataloader, val_dataloader ,criterion, optimizer, num_epochs, scheduler, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    Model_Size = count_parameters(model)

    # print(f"Total parameters in FeatureExtractor: {Low_Level_Size}")
    # print(f"Total parameters in high level: {High_Level_Size}")
    print(f"Total parameters in the entire model: {Model_Size}")


   


    lr_warmup_step=5
    epoch_time=0.0
    # Training loop
    best_valloss=1000
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        loss_acumulated = 0.0

        if epoch < lr_warmup_step:
            warmup_factor = (epoch + 1) / lr_warmup_step
            optimizer.param_groups[0]['lr'] = lr * warmup_factor
        
        model.train()
        optimizer.zero_grad()

        for batch in train_dataloader:

            padded_inputs,  labels  = batch
            # Move data to the correct device
            padded_inputs = padded_inputs.to(device)
            # padded_speed_inputs=padded_speed_inputs.to(device)
            labels = labels.to(device)
            # lengths = lengths.to(device)

            
            with torch.set_grad_enabled(True):
                outputs = model(padded_inputs)
                # print(outputs.shape)
                # print(labels.shape)
                # print(outputs.min(), outputs.max())  # logits or probabilities
                # print(labels.unique()) 
                

                loss = criterion(outputs, labels.float())
                epoch_loss += loss.item()
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)
                optimizer.step()  # Update weights
                optimizer.zero_grad()

        
        epoch_loss = (epoch_loss/len(train_dataloader))
        
        
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_time+=epoch_duration
        
        
        validation=1
        val_loss = evaluate_model(validation, model, val_dataloader, criterion)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, val_Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds')
        

        if val_loss < best_valloss:
            
            best_valloss = val_loss
            print("####Ding ding ding! We found a new best model!")
            best_model_state = copy.deepcopy(model.state_dict())

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({'train_loss': epoch_loss, 'epoch': epoch})
        wandb.log({'val_loss': val_loss, 'epoch': epoch})
        wandb.log({'train_time': epoch_time, 'epoch': epoch })
        wandb.log({'Learning_rate': current_lr, 'epoch': epoch })
    
    # # Load the best state dictionary back into the model before returning
    # if best_model_state is not None:
    #     model.load_state_dict(best_model_state)

    return model, best_valloss, best_model_state





# SECTION 4: Evaluation Function
def evaluate_model(validation, model, test_dataloader, criterion):
    

    predictions=[]
    predictions2=[]

    total_loss = 0.0
    total_loss2 = 0.0
    total_mape = 0.0
    total_mape0 = 0.0
    total_mape1 = 0.0
    total_mape2 = 0.0
    total_mape02 = 0.0
    total_mape12 = 0.0
    total_r2 = 0.0
    total_r2_0 = 0.0
    total_r2_1 =0.0
    total_r22 = 0.0
    total_r2_02 = 0.0
    total_r2_12 =0.0
    test_ttime=0.0
    test_ttime2=0.0
    model.eval()
    if validation == 1:
        with torch.no_grad():
            for batch in test_dataloader:

                inputs, labels  = batch
                # Move data to the correct device
                inputs = inputs.to(device)
                # inputs2 = inputs2.to(device)
                labels = labels.to(device)
                # lengths = lengths.to(device)
                
                outputs = model(inputs)

                
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            val_loss = total_loss / len(test_dataloader)
        return val_loss

    elif validation == 0:          
        with torch.no_grad():
            start_time = time.time()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            total_tp, total_fp, total_fn = 0, 0, 0
            num_batches = 0
            for batch in test_dataloader:

                inputs, labels  = batch
                # Move data to the correct device
                inputs = inputs.to(device)
                # inputs2 = inputs2.to(device)
                labels = labels.to(device)
                # lengths = lengths.to(device)

                
                

                outputs = model(inputs)


                print("Test_ Output stats:", outputs.min().item(), outputs.max().item())
                print("Test_ Labels stats:", labels.min().item(), labels.max().item())
                print("Sigmoid(Outputs) mean Test set:", torch.sigmoid(outputs).mean().item())
                preds = (torch.sigmoid(outputs) > 0.5).float()
                acc = (preds == labels).float().mean().item()
                print("test Batch acc:", acc)


                
                             

                
                # Flatten before loss
                loss = criterion(outputs.view(-1), labels.view(-1))
                total_loss += loss.item()
                
                # Probabilities and binary predictions
                preds = torch.sigmoid(outputs)
                preds_binary = (preds > 0.5).float()

                # Element-wise accuracy
                correct = (preds_binary == labels).float().sum()
                total_correct += correct.item()
                total_samples += labels.numel()

                # Precision/recall/F1 counters
                tp = ((preds_binary == 1) & (labels == 1)).sum().item()
                fp = ((preds_binary == 1) & (labels == 0)).sum().item()
                fn = ((preds_binary == 0) & (labels == 1)).sum().item()
                total_tp += tp
                total_fp += fp
                total_fn += fn

                num_batches += 1

                predictions.append([(labels, preds_binary)])

                end_time = time.time()
                epoch_duration = end_time - start_time
                test_ttime += epoch_duration
                start_time = time.time()
                

        test_loss = total_loss / num_batches
        accuracy = total_correct / total_samples
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        test_time_tot = test_ttime  

        test_time_ave = test_ttime / len(test_dataloader.dataset)

        return test_loss, accuracy, precision, recall, f1, test_time_tot, test_time_ave


def run_hyperparameter_sweep():

    sweep_configuration = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'batch_size': {'values': [32, 100]},
            'learning_rate': {'values': [0.00001, 0.00005, 0.0001]},
            'weight_decay': {'values': [0, 5e-5]},
            'conv1': {'values': [64, 32]},
            'conv2': {'values': [96, 64]}, 
            'conv1_out_channels': {'values': [98, 128, 160]},
            'conv2_out_channels': {'values': [128, 160, 256]},  
            'conv3_out_channels': {'values': [160, 256]},
            'conv4_out_channels': {'values': [256, 380, 512]},
            'drop_out': {'values': [0, 0.3]},
            },
            'early_terminate': {
        'type': 'hyperband',
        'min_iter': 15}
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project=f'{filename}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return sweep_id



# Define the paths for the model and the best score


model_save_path = f"/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Benchmarks/weightandlogs/{filename}_best_model.pt"
best_score_path = f"/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Benchmarks/weightandlogs/{filename}_best_best_mape.txt"
best_logs_metrics = f"/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Benchmarks/weightandlogs/{filename}_best_log"


# full_model_save_path = "best_full_model.pt"

def _get_best_mape():
    """
    Reads the best MAPE from a file to persist the score across runs.
    Returns a high value if the file does not exist.
    """
    if os.path.exists(best_score_path):
        with open(best_score_path, "r") as f:
            try:
                return float(f.read().strip())
            except Exception:
                print("##### Warning: best_score_path corrupted, resetting to 100.0")
                return 0.01
    return 0.01


best_avg_mape = _get_best_mape()

def sweep_train():
    # Initialize the global variable with the best score from the file, if available.
    global best_avg_mape   
    with wandb.init(mode = "offline"):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = GoogLeNetV1(num_classes=30, aux_logits=False).to(device)

        Model_Size = count_parameters(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=6, verbose=True
        )

        # criterion = nn.MSELoss()
        criterion = nn.BCEWithLogitsLoss()

        # Usage
        set_seeds(0)
        train_dataloader, val_dataloader, test_dataloader = Data_loaders(config.batch_size, force=False)



        # Train and validate
        model, val_loss, best_model_states = train_and_val(
            model, train_dataloader, val_dataloader,
            criterion, optimizer, num_epochs=80, scheduler=scheduler,lr=config.learning_rate
        )

        # Evaluate on test set

        model.load_state_dict(best_model_states)


        validation = 0
        test_loss, accuracy, avg_precision, avg_recall, avg_f1, test_time_tot, test_time_ave= evaluate_model(
            validation, model, test_dataloader, criterion
        )

        # Log test results
        wandb.log({
            'test_loss': test_loss,
            'accuracy': accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'test_time_tot': test_time_tot,
            'test_time_ave': test_time_ave,
            'parameter_count': Model_Size
        })
        print(f'test_accuracy: {accuracy}')
       

        

        if accuracy > best_avg_mape:
            best_avg_mape = accuracy
            print(f"🔥 New best MAPE found: {best_avg_mape:.4f}")

            # Save model
            torch.save(copy.deepcopy(model.state_dict()), model_save_path)

            # Save best score (just number for reloading)
            with open(best_score_path, "w") as f:
                f.write(str(best_avg_mape))

            # Save logs/config/metrics for inspection
            best_metrics = {
                'test_loss': test_loss,
                'accuracy': accuracy,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1,
                'test_time_tot': test_time_tot,
                'test_time_ave': test_time_ave, 
                'parameter_count': Model_Size,
                'config': config.as_dict(),
                'wandb summary':dict(wandb.run.summary)
            }


            with open(best_logs_metrics, "w") as f:
                json.dump(best_metrics, f, indent=4)
            

# ---- RUN SWEEP ----
sweep_id = run_hyperparameter_sweep()

wandb.agent(sweep_id, function=sweep_train, count=100)


