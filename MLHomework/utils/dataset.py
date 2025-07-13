import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FEATURES = ['Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
            'Sub_metering_remainder', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

def compute_target(df):
    return df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_remainder']].sum(axis=1)

class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X, y, input_len=90, pred_len=90):
    Xs, ys = [], []
    for i in range(len(X) - input_len - pred_len + 1):
        Xs.append(X[i:i+input_len])
        ys.append(y[i+input_len:i+input_len+pred_len].flatten())
    return np.array(Xs), np.array(ys)

def load_data(train_path, test_path, input_len=90, pred_len=90):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_target = compute_target(train_df).values.reshape(-1, 1)
    test_target = compute_target(test_df).values.reshape(-1, 1)

    sx = StandardScaler()
    sy = StandardScaler()
    X_train = sx.fit_transform(train_df[FEATURES])
    X_test = sx.transform(test_df[FEATURES])
    y_train = sy.fit_transform(train_target)
    y_test = sy.transform(test_target)

    X_seq_train, y_seq_train = create_sequences(X_train, y_train, input_len, pred_len)
    X_seq_test, y_seq_test = create_sequences(X_test, y_test, input_len, pred_len)

    return PowerDataset(X_seq_train, y_seq_train), PowerDataset(X_seq_test, y_seq_test), sy
