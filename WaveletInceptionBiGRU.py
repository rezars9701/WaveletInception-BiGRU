"""
Created on Mon Dec 20 2025

@author: rriahisamani
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.transforms import Resize
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import json
import wandb
import pickle
import os
import copy
import random
from scipy.io import loadmat
from utils import NeuralDWAV
import matplotlib.cm as cm
import torch.nn.functional as F
import os
from scipy.signal import butter, filtfilt
from utils.cwt import CWT
from typing import Optional
import torch




filename = os.path.splitext(os.path.basename(__file__))[0]

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU-based operations
    np.random.seed(seed)
    
set_seeds(0)


class VibrationAugment:
    def __init__(self,
                 noise_std=0.05,
                 scale_range=(0.9, 1.1),
                 jitter_prob=0.5,
                 crop_prob=0.5,
                 crop_frac=0.9,
                 freq_mask_prob=0.1,
                 freq_mask_size=4):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.jitter_prob = jitter_prob
        self.crop_prob = crop_prob
        self.crop_frac = crop_frac
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_size = freq_mask_size

    def __call__(self, x):

        # Add Gaussian noise
        if random.random() < 0.5:
            x = x + self.noise_std * torch.randn_like(x)

        # Amplitude scaling
        if random.random() < 0.5:
            factor = torch.empty(1).uniform_(*self.scale_range).item()
            x = x * factor

        # Jitter (small random time shifts)
        if random.random() < self.jitter_prob:
            shift = random.randint(-5, 5)  # shift up to ±5 samples
            x = torch.roll(x, shifts=shift, dims=-1)

        # Random crop (keep only a fraction of signal, then pad back)
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
        self.augment = VibrationAugment(noise_std=0.10,
                 scale_range=(0.8, 1.2),
                 jitter_prob=0.5,
                 crop_prob=0.2,
                 crop_frac=0.9,
                 freq_mask_prob=0.5,
                 freq_mask_size=5) 

    def __len__(self):
        return len(self.input_sequences_1)

    def __getitem__(self, idx):

        x1 = torch.tensor(self.input_sequences_1[idx], dtype=torch.float32)
        x2 = torch.tensor(self.input_sequences_2[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.augment is not None:
            x1 = self.augment(x1)
            # x2 = self.augment(x2)

        return x1, x2, y
        

def aba_segmentation(dataset_list):
    input_arrays = []
    input_sppeds = []
    for array1, array2,  _, _ in dataset_list:
        input_arrays.append(array1)
        input_sppeds.append(array2)
        
    return input_arrays, input_sppeds




def preprocess_and_save(force=False):
    """Preprocess dataset once and save to .pt file"""
    save_path = "/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Datasets/preprocessed_dataset_romania2.pt"

    if os.path.exists(save_path) and not force:
        print(f"Loading cached preprocessed dataset from {save_path}")
        return torch.load(save_path)

    print("Preprocessing dataset from scratch...")
    


    data = loadmat('/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Datasets/ALL_sig_struct_part11.mat', squeeze_me=True)
    all_sig1 = data['ALL_sig_struct1']
    data = loadmat('/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Datasets/ALL_sig_struct_part22.mat', squeeze_me=True)
    all_sig2 = data['ALL_sig_struct2']

    all_sig = np.concatenate((all_sig1, all_sig2))


    signals = all_sig['Signal']   # numpy array, variable length
    labels  = all_sig['Label']    # 30x1 array
    dis_rngs = all_sig['dis_rng'] # 1x2 array
    speeds = all_sig['Speed']    # 30x1 array


    data = loadmat('/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Datasets/ALL_sig_struct33_tes.mat', squeeze_me=True)

    all_sig = data['ALL_sig_struct3']

    signal_test = all_sig['Signal']   # numpy array, variable length
    label_test  = all_sig['Label']    # 30x1 array
    dis_rng_test = all_sig['dis_rng'] # 1x2 array
    speed_test = all_sig['Speed']    # 30x1 array


    all_signals, all_labels1, all_rng, all_speeds = [], [], [], []

    for i in range(len(signals)):
        signal = np.array(signals[i])
        label = np.array(labels[i])
        rang = np.array(dis_rngs[i])
        speed = np.array(speeds[i])

        # Convert to torch
        signal_t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # shape [1, L]
        label_t = torch.tensor(label, dtype=torch.float32).unsqueeze(1)
        rang_t = torch.tensor(rang, dtype=torch.float32).unsqueeze(0)
        speed_t = torch.tensor(speed, dtype=torch.float32).unsqueeze(0)

        all_signals.append(signal_t)
        all_labels1.append(label_t)
        all_rng.append(rang_t)  # if second label same or separate
        all_speeds.append(speed_t)


    print('length dataset', len(all_signals))
    # Construct tuples compatible with your existing pipeline
    full_dataset = list(zip(all_signals, all_speeds, all_labels1, all_rng))

    print('length dataset2all_speeds', len(all_speeds))

    all_signals_tst, all_labels1_tst, all_rng_tst, all_speeds_tst = [], [], [], []

    for i in range(len(signal_test)):
        signal = np.array(signal_test[i])
        label = np.array(label_test[i])
        rang = np.array(dis_rng_test[i])
        speed = np.array(speed_test[i])

        # Convert to torch
        signal_t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0) # shape [1, L]
        label_t = torch.tensor(label, dtype=torch.float32).unsqueeze(1)
        rang_t = torch.tensor(rang, dtype=torch.float32).unsqueeze(0)
        speed_t = torch.tensor(speed, dtype=torch.float32).unsqueeze(0)

        all_signals_tst.append(signal_t)
        all_labels1_tst.append(label_t)
        all_rng_tst.append(rang_t)  # if second label same or separate
        all_speeds_tst.append(speed_t)


    # Construct tuples compatible with your existing pipeline
    full_dataset_test = list(zip(all_signals_tst, all_speeds_tst, all_labels1_tst, all_rng_tst))

    total_size = len(full_dataset)
    train_size = int(0.85 * total_size)
    val_size = int(0.15 * total_size)
    

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    test_dataset=full_dataset_test

    # === TRAIN ===

    # === TRAIN ===dataset Normaliztiona
    input_arrays_training, input_arrays_training_speed = aba_segmentation(train_dataset)

    # print('input_arrays_training_speed', input_arrays_training_speed)
    flattened_tensors = [t.view(-1) for t in input_arrays_training]
    input_aba = torch.cat(flattened_tensors)

    input_mean = input_aba.mean().item()
    input_std = input_aba.std().item()

    train_sequences = [(t - input_mean) / (input_std+ 1e-8) for t in input_arrays_training]


    flattened_tensors_s = [t.view(-1) for t in input_arrays_training_speed]
    input_speed_tr = torch.cat(flattened_tensors_s)


    labels_values = [torch.tensor(array2) for _, _, array2, _ in train_dataset]
    train_labels = torch.stack(labels_values)


    # === VAL ===
    val_input, val_input_s = aba_segmentation(val_dataset)
    val_sequences = [(t - input_mean) / (input_std+ 1e-8) for t in val_input]
    # val_sequences_speed = [(t - input_mean_s) / (input_std_s+ 1e-8) for t in val_input_s]
    flattened_tensors_s = [t.view(-1) for t in val_input_s]
    input_speed_vl = torch.cat(flattened_tensors_s)

    labels_values = [torch.tensor(array2) for _,_, array2,_ in val_dataset]
    val_labels = torch.stack(labels_values)
    
    

    # === TEST ===
    test_input, test_input_s = aba_segmentation(test_dataset)
    test_sequences = [(t - input_mean) / (input_std+ 1e-8) for t in test_input]
    # test_sequences_speed = [(t - input_mean_s) / (input_std_s+ 1e-8) for t in test_input_s]
    flattened_tensors_s = [t.view(-1) for t in test_input_s]
    input_speed_test = torch.cat(flattened_tensors_s)
    labels_values = [torch.tensor(array2) for _, _, array2,_ in test_dataset]
    
    test_labels = torch.stack(labels_values)


    # Save everything
    data_dict = {
        "train_sequences": train_sequences,
        "train_sequences_speed":input_speed_tr,
        "train_labels": train_labels,
        "val_sequences": val_sequences,
        "val_sequences_speed": input_speed_vl,
        "val_labels": val_labels,
        "test_sequences": test_sequences,
        "test_sequences_speed": input_speed_test,
        "test_labels": test_labels,
    }
    torch.save(data_dict, save_path)
    print(f"Saved preprocessed dataset to {save_path}")
    return data_dict


def butter_lowpass_filter(signal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal



def Data_loaders(bs,hope_size, force=False):

    data_dict = preprocess_and_save(force=force)
    fs = 25600
    cutoff = 100 
    order = 4 
    def collate_fn(batch, resize_dim=(274, 274)):

        inputs, inputs_s, labels = zip(*batch)
        lengths = torch.tensor([seq.size(1) for seq in inputs])
        sorted_indices = torch.argsort(lengths, descending=True)

        inputs = [inputs[i] for i in sorted_indices]
        inputs_s = [inputs_s[i] for i in sorted_indices]
        labels = torch.stack([labels[i] for i in sorted_indices])

        # max_len = lengths.max().item()
        max_len = 78150
        iput_tens = []
        for seq in inputs:
                       
            seq = seq.squeeze().cpu().numpy()  # convert to NumPy for filtering
            seq = butter_lowpass_filter(seq, cutoff, fs, order)
            seq = np.copy(seq)
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            iput_tens.append(seq.float())

        padded_inputs = torch.stack([torch.nn.functional.pad(seq, (0, 78150- seq.shape[1]))
                                     for seq in iput_tens])


        inputs_s = torch.stack(inputs_s).unsqueeze(1)


        lens = lengths[sorted_indices]

        masks = [torch.cat([torch.ones(l), torch.zeros(max_len - l)]) for l in lens]
        masks = torch.stack(masks)   # (B, max_len)
        return padded_inputs, inputs_s, labels, masks, lens

    train_dataset = CustomDataset(data_dict["train_sequences"],data_dict["train_sequences_speed"], data_dict["train_labels"],augment=True)
    val_dataset = CustomDataset(data_dict["val_sequences"],data_dict["val_sequences_speed"], data_dict["val_labels"])
    test_dataset = CustomDataset(data_dict["test_sequences"], data_dict["test_sequences_speed"], data_dict["test_labels"])

    train_dataloader = DataLoader(train_dataset, batch_size=bs, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, collate_fn=collate_fn, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader





#model defination 



class _ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1,dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, stride=stride)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int = 32, 
        use_residual: bool = True,
        scaling: float = 0.1,         
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.scaling = scaling
        
        self.branch1 = _ConvBNAct(in_channels, bottleneck_channels, kernel_size=1)


        self.branch2 = nn.Sequential(
            _ConvBNAct(in_channels, bottleneck_channels, kernel_size=1),
            _ConvBNAct(bottleneck_channels, bottleneck_channels, kernel_size=3)
        )

        self.branch3 = nn.Sequential(
            _ConvBNAct(in_channels, bottleneck_channels, kernel_size=1),
            _ConvBNAct(bottleneck_channels, bottleneck_channels, kernel_size=3),
            _ConvBNAct(bottleneck_channels, bottleneck_channels, kernel_size=3)
        )

        concat_channels = bottleneck_channels * 3
        self.project = nn.Sequential(
            nn.Conv1d(concat_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )

        # Residual projection handles channel mismatch (e.g., 98 to 128)
        if self.use_residual:
            if in_channels != out_channels:
                self.residual_proj = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels)
                )
            else:
                self.residual_proj = nn.Identity()

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        
        # Parallel Processing
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Concatenate and Project
        x_cat = torch.cat([b1, b2, b3], dim=1)
        x_proj = self.project(x_cat)
        x_proj = self.dropout(x_proj)

        if self.use_residual:
            # Apply Residual Scaling (x_proj * scale + shortcut)
            res = self.residual_proj(residual)
            out = self.act(res + (x_proj * self.scaling))
        else:
            out = self.act(x_proj)

        return out


def compute_output_lengths(lengths, layers, wavelet_level=None):
    if wavelet_level is not None:
        lengths = torch.floor(lengths / (2 ** wavelet_level))

    for layer in layers:
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.MaxPool1d):
            kernel = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            stride = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            padding = layer.padding if isinstance(layer.padding, int) else layer.padding[0]
            dilation = layer.dilation if isinstance(layer.dilation, int) else layer.dilation[0]

            lengths = torch.floor(
                (lengths + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
            )
    return lengths.to(torch.int64)


class WaveletInceptionBiGRU(nn.Module):
    def __init__(self, level, fr, conv1, conv2,Inc_layers_config, lstm_hidden_sizes, dropout, speed_dens,num_classes=20):
        super(WaveletInceptionBiGRU, self).__init__()


        self.speed_fc = nn.Sequential(nn.Linear(1, speed_dens//2),  # Transform speed into a 16-dimensional vector
            nn.ReLU(),
            nn.Linear(speed_dens//2, speed_dens),  # Transform into a 32-dimensional vector
            nn.ReLU())

        self.maxlength=78150


        self.stem = NeuralDWAV.NeuralDWAV(1,Input_Level=level,Input_Archi="WPT",Filt_Trans = False,
                                            Filt_Mother = "db4",Act_Train = True ).float()


        inp=2**level

        self.projection=_ConvBNAct(inp, conv1, kernel_size=1)

        self.conv1 = _ConvBNAct(conv1, conv1, kernel_size=7, stride=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = _ConvBNAct(conv1, conv2, kernel_size=5)


        self.Inc_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        input_channels=conv2

        # Create convolutional and pooling layers based on the configuration
        for out_channels in Inc_layers_config:
            current_bottleneck = out_channels // 4
            self.Inc_layers.append(InceptionModule(in_channels=input_channels,
        out_channels=out_channels, bottleneck_channels=current_bottleneck, use_residual= True))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=3, stride=2))
            input_channels =  out_channels # Update input channels for the next layer

        input_size = input_channels + speed_dens
        
        self.BiGRU1 = nn.GRU(input_size=input_size, hidden_size=lstm_hidden_sizes[0], num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        
        self.BiGRU2 = nn.GRU(lstm_hidden_sizes[0]*2 , lstm_hidden_sizes[1],num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.drop_out=nn.Dropout(dropout)

        # input_size = input_channels + speed_dens
        
        self.fc2 = nn.Sequential(nn.Linear(lstm_hidden_sizes[1]*2, lstm_hidden_sizes[1]//2),  # Transform speed into a 16-dimensional vector
            nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(lstm_hidden_sizes[1]//2, 1))

    def forward(self, x,speed,masks, lens):

        
        x=x.to(device)
        print('shape 1 ', x.shape)
        x= self.stem(x)
        x = torch.stack(x, dim=0)
        x = x.squeeze(2)
        x = x.permute(1,0,2)

        print('shape 2 ', x.shape)
        x = self.projection(x)

        print('shape 3 ', x.shape)


        x = self.conv1(x)
        print('shape 4 ', x.shape)
        # x = self.maxpool1(x)
        x = self.conv2(x)

        print('shape 5 ', x.shape)

        x=self.Inc_layers[0](x)

        print('shape 6', x.shape)

        x=self.Inc_layers[1](x)
        print('shape 7 ', x.shape)
        

        x=self.pool_layers[0](x)
        print('shape 8 ', x.shape)
        x=self.Inc_layers[2](x)

        print('shape 9 ', x.shape)
        
        
        x=self.Inc_layers[3](x)
        print('shape 10 ', x.shape)
        x=self.pool_layers[0](x)
        print('shape 11 ', x.shape)
        # print('input_shape_9', x.size())

        # print('speed shape', speed.size())

        targetlen = x.size(2)
        speed = speed.unsqueeze(2).expand(-1, -1, targetlen)
        print('speed shape2', speed.size())

        # print('mask shape', masks.size())

        pool = nn.AdaptiveMaxPool1d(targetlen)
        # print('speed sahpe 3')
        masks = pool(masks.unsqueeze(1).float())
        print('mask shape3', masks.size())
        # print('masks', masks)

        speed = speed * masks

        speed = speed.permute(0, 2, 1).to(device)  # (B, L, C)
        print('speed shape3.5', speed.size())
        speed = self.speed_fc(speed)
        print('speed shape4.5', speed.size())
        speed = speed.permute(0, 2, 1)  # (B, C, L)
        print('speed shape5', speed.size())
        x = torch.cat((x, speed), dim=1)  # Concatenate along channel dimension
        print('input_shape_12.5', x.size())
        x= x.permute(0, 2, 1)  # (B, L, C)
        print('input_shape_13', x.size())
        x, _ = self.BiGRU1(x)
        print('input_shape_14', x.size())
        #print('input_shape_9.9', x.size())
        seq_size=x.size(1)
        step_size = seq_size // 30
        start_point = step_size // 2
        selected_steps = torch.arange(start_point, seq_size, step_size)[:30].long()
        x = x[:,selected_steps,:]
        print('shape 15 ', x.shape)
        x, _ = self.BiGRU2(x)
        print('shape 16 ', x.shape)
        x = self.drop_out(x)
        print('input_shape_17', x.size())
        x = self.fc2(x)
        print('input_shape_18', x.size())

        d
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
    training_log = []

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

            padded_inputs, speed,  labels, mask, lengths  = batch

            labels = labels.to(device)
            lengths = lengths.to(device)

            
            with torch.set_grad_enabled(True):
                outputs = model(padded_inputs, speed,mask, lengths)
                loss = criterion(outputs.view(-1), labels.float().view(-1))
                epoch_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.9)
                optimizer.step()
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

                inputs, speed, labels, mask, lengths  = batch
                
                labels = labels.to(device)
                lengths = lengths.to(device)
                
                outputs = model(inputs, speed, mask, lengths)
  
                loss = criterion(outputs.view(-1), labels.view(-1))
                total_loss += loss.item()

            val_loss = total_loss / len(test_dataloader)
        return val_loss

    elif validation == 0:          
        with torch.no_grad():
            start_time = time.time()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
            num_batches = 0
            print("Starting testing...")
            print("Number of test batches:", len(test_dataloader.dataset))
            for batch in test_dataloader:

                inputs, speed, labels, mask, lengths  = batch

                labels = labels.to(device)
                lengths = lengths.to(device)
                outputs = model(inputs, speed, mask, lengths)
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

                tp = ((preds_binary == 1) & (labels == 1)).sum().item()
                fp = ((preds_binary == 1) & (labels == 0)).sum().item()
                fn = ((preds_binary == 0) & (labels == 1)).sum().item()
                tn = ((preds_binary == 0) & (labels == 0)).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

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

        precision_1 = total_tp / (total_tp + total_fp + 1e-8)
        recall_1    = total_tp / (total_tp + total_fn + 1e-8)
        f1_1        = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-8)

        precision_0 = total_tn / (total_tn + total_fn + 1e-8)
        recall_0    = total_tn / (total_tn + total_fp + 1e-8)
        f1_0        = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + 1e-8)

        return test_loss, accuracy, precision, recall, f1, test_time_tot, test_time_ave, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, predictions


def run_hyperparameter_sweep():

    sweep_configuration = {
        'method': 'random',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'batch_size': {'values': [32]},
            'learning_rate': {'values': [0.00015]},
            'weight_decay': {'values': [0]},
            'level': {'values': [7]},
            'conv1': {'values': [96]},
            'conv2': {'values': [96]}, 
            'fr': {'values': [128]},
            'conv1_out_channels': {'values': [128]},
            'conv2_out_channels': {'values': [128]},  
            'conv3_out_channels': {'values': [256]},
            'conv4_out_channels': {'values': [380]},
            'drop_out': {'values': [0.3]},
            'speed_dens':{'values': [64]},
            'lstm_hidden_size_1': {'values': [128]},
            'lstm_hidden_size_2': {'values': [96]},
            'hope': {'values': [0]},
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


model_save_path = f"/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Wights_logs/{filename}_best_model.pt"
best_score_path = f"/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Wights_logs/{filename}_best_best_mape.txt"
best_logs_metrics = f"/home/rriahisamani/Python/2_Feature_extraction/models/WaveletInceptionNetwork/Romania/Wights_logs/{filename}_best_log"


# full_model_save_path = "best_full_model.pt"

def _get_best_acc():
    if os.path.exists(best_score_path):
        with open(best_score_path, "r") as f:
            try:
                return float(f.read().strip())
            except Exception:
                print("##### Warning: best_score_path corrupted, resetting to 100.0")
                return 0.01
    return 0.01


def compute_dataset_stats(dataloader, name="dataset"):
    mean_list, std_list, min_list, max_list = [], [], [], []
    for i, batch in enumerate(dataloader):
        inputs, speed, labels, lengths = batch
        # Flatten input to compute simple statistics
        mean_list.append(inputs.mean().item())
        std_list.append(inputs.std().item())
        min_list.append(inputs.min().item())
        max_list.append(inputs.max().item())
        if i == 10:  # check only first 10 batches to save time
            break
    print(f"\n{name} input stats → mean: {torch.tensor(mean_list).mean():.4f}, "
          f"std: {torch.tensor(std_list).mean():.4f}, "
          f"min: {torch.tensor(min_list).mean():.4f}, "
          f"max: {torch.tensor(max_list).mean():.4f}")


best_avg_mape = _get_best_acc()

def sweep_train():
    # Initialize the global variable with the best score from the file, if available.
    global best_avg_mape   
    with wandb.init(mode="offline"):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        Inc_layers_config = [config.conv1_out_channels]
        if config.conv2_out_channels is not None:
            Inc_layers_config.append(config.conv2_out_channels)
        if config.conv3_out_channels is not None:
            Inc_layers_config.append(config.conv3_out_channels)
        if config.conv4_out_channels is not None:
            Inc_layers_config.append(config.conv4_out_channels)

        
        lstm_hidden_sizes = [config.lstm_hidden_size_1]
        if config.lstm_hidden_size_2 is not None:
            # lstm_2 = min(config.lstm_hidden_size_1, config.lstm_hidden_size_2)
            lstm_hidden_sizes.append(config.lstm_hidden_size_2)


        model = WaveletInceptionBiGRU(level=config.level,fr=config.fr, conv1=config.conv1,
            conv2=config.conv2,
            Inc_layers_config=Inc_layers_config,lstm_hidden_sizes=lstm_hidden_sizes, 
            dropout=config.drop_out,speed_dens=config.speed_dens
        ).to(device)

        Model_Size = count_parameters(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=6
        )

        criterion = nn.BCEWithLogitsLoss()

        train_dataloader, val_dataloader, test_dataloader  = Data_loaders(config.batch_size,config.hope, force=False)


        # Train and validate
        model, val_loss, best_model_states= train_and_val(
            model, train_dataloader, val_dataloader,
            criterion, optimizer, num_epochs=80, scheduler=scheduler, lr=config.learning_rate
        )
        # state_dict = torch.load(model_save_path, map_location=device)
        # model.load_state_dict(state_dict, strict=True)
        model.load_state_dict(best_model_states, strict=True)

        validation = 0
        test_loss, accuracy, avg_precision, avg_recall, avg_f1, test_time_tot, test_time_ave, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, predictions= evaluate_model(validation, model, test_dataloader, criterion
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
            'parameter_count': Model_Size,
            'precision_0': precision_0,
            'recall_0': recall_0,
            'f1_0': f1_0,
            'precision_1': precision_1,
            'recall_1': recall_1,
            'f1_1': f1_1,
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

            # with open(best_logs, "w") as f:
            #     f.write(json.dumps(best_metrics, indent=4))
            with open(best_logs_metrics, "w") as f:
                json.dump(best_metrics, f, indent=4)


# ---- RUN SWEEP ----
sweep_id = run_hyperparameter_sweep()

wandb.agent(sweep_id, function=sweep_train, count=15)


