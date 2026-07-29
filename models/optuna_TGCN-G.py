import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, ChebConv

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import time
import matplotlib.pyplot as plt

import optuna

cmd = 'C:/Users/LSY/Graduation Paper/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Feature Data
feature_np = np.load(cmd + '018_feature_cg6+cr.npy')
feature_np.shape

feature_tensor = torch.tensor(feature_np, dtype = torch.float).to(device)
feature_tensor.size()


# Edge index & Adjacent matrix
edge_index_np = np.load(cmd + '019_edge_index.npy')
edge_index = torch.tensor(edge_index_np).to(device)
print(edge_index.shape)   # (2, 9358)

reciprocal_matrix = np.load(cmd + '020_reciprocal_matrix.npy')
edge_weight_np = reciprocal_matrix[edge_index_np[0], edge_index_np[1]].reshape(-1, 1)
edge_weight = torch.tensor(edge_weight_np, dtype=torch.float).to(device)
print(edge_weight.shape)   # (9358, 1)


class TGCN_encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TGCN_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.create_params_layers()
    
    def create_update_gate(self):
        self.conv_z = GCNConv(in_channels = self.in_channels, out_channels = self.out_channels)
        self.linear_z = nn.Linear(2 * self.out_channels, self.out_channels)
    
    def create_reset_gate(self):
        self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels)
        self.linear_r = nn.Linear(2 * self.out_channels, self.out_channels)

    def create_candidate_state(self):
        self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels)
        self.linear_h = nn.Linear(2 * self.out_channels, self.out_channels)

    def create_params_layers(self):
        self.create_update_gate()
        self.create_reset_gate()
        self.create_candidate_state()

    def set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H
    
    def calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis = 1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z
    
    def calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_h(X, edge_index, edge_weight), H], axis = 1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R
    
    def calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis = 1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde
    
    def calcutlate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H
    
    def forward(self, X, edge_index, edge_weight, H):
        H = self.set_hidden_state(X, H)
        Z = self.calculate_update_gate(X, edge_index, edge_weight, H)
        R = self.calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self.calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self.calcutlate_hidden_state(Z, H, H_tilde)
        return H
    

class TGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TGCN, self).__init__()
        self.tgcn_encoder = TGCN_encoder(in_channels, out_channels)

        # Output layer for 10, 20, 30min prediction
        self.fc = nn.Linear(out_channels, 1)
    
    def forward(self, X, edge_index, edge_weight, H):
        h_0 = self.tgcn_encoder(X, edge_index, edge_weight, H)
        h = F.relu(h_0)

        out = self.fc(h)

        return out, h


# find best hyperparameter set
def objective(trial):
    # Hyperparameter search space
    out_channels = trial.suggest_categorical("out_channels", [16, 32, 64, 128, 256])
    epochs = trial.suggest_categorical("epoch", [50, 100])
    
    # Initialize model, optimizer, and loss function
    model = TGCN(in_channels, out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train Model
    model.train()
    learning_loss = []
    for epoch in range(epochs):
        total_cost = 0
        H = None
        for t, snapshot in enumerate(train):
            x = snapshot.x
            y = snapshot.y

            out, h = model(x, edge_index, edge_weight, H)
            cost = criterion(out, y)

            total_cost += cost.item()

        # Average cost
        avg_cost = total_cost / (t + 1)
        learning_loss.append(avg_cost)

        optimizer.zero_grad()   # 기울기 초기화
        cost.backward()   # 역전파
        optimizer.step()   # 최적화
    
    return avg_cost

in_channels = 7
lr = 0.001



min_list = [10, 20, 30]
min_mse, min_mae = [], []
for min in min_list:
    target_np = np.load(cmd + '018_target' + str(min) + '_nnor.npy')
    target_tensor = torch.tensor(target_np, dtype = torch.float).to(device)

    database = []
    for i in tqdm(range(len(feature_tensor))):
        data = Data(
            x = feature_tensor[i],
            y = target_tensor[i]
        )
        database.append(data)

    # Split Test, Train data
    split_ratio = 0.7
    split_idx = int(len(database) * split_ratio)

    # split_idx = 4172
    train, test = database[:split_idx], database[split_idx:]

    # hyperparameter
    # in_channel = 7
    # out_channel = 128


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    # txt 파일로 저장
    with open(cmd + '024_TGCN-G_test_result2/best_params'+ str(min) + '.txt', "w") as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    model = TGCN( 
        in_channels, 
        study.best_params['out_channels']
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Train Model
    model.train()
    learning_loss = []
    for epoch in range(study.best_params['epoch']):
        total_cost = 0
        H = None
        s_time = time.time()
        for t, snapshot in enumerate(train):
            x = snapshot.x
            y = snapshot.y

            out, h = model(x, edge_index, edge_weight, H)
            cost = criterion(out, y)

            total_cost += cost.item()

        # Average cost
        avg_cost = total_cost / (t + 1)
        learning_loss.append(avg_cost)

        optimizer.zero_grad()   # 기울기 초기화
        cost.backward()   # 역전파
        optimizer.step()   # 최적화

        e_time = time.time()

        print(f"Epoch {epoch} train MSE: {cost: .4f}")
        print(f"Elapsed Time: {e_time - s_time: .4f}")




    # Make Train Loss Graph
    plt.figure(figsize=(16,8))
    plt.plot(learning_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(cmd + '024_TGCN-G_test_result2/024_TGCN-G_loss_graph' + str(min) + '.jpg', dpi = 600)
    # plt.show() 


    # Evaluate Model
    model.eval()
    mae_criterion = nn.L1Loss()
    total_cost = 0
    total_mae = 0
    predictions, actuals = [], []
    with torch.no_grad():
        for t, snapshot in enumerate(tqdm(test)):
            x = snapshot.x
            y = snapshot.y

            out, h = model(x, edge_index, edge_weight, H)

            cost = criterion(out, y)
            mae = mae_criterion(out, y)
            
            total_cost += cost.item()
            total_mae += mae.item()

            predictions.append(out.detach().cpu().numpy())
            actuals.append(y.cpu().numpy())

        cost = total_cost / (t+1)
        mae = total_mae / (t+1)

    print("MSE: {:.4f}".format(cost))
    print("MAE: {:.4f}".format(mae))

    min_mse.append(cost.item())
    min_mae.append(mae)

    # 실제 값과 예측 값 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(np.array(actuals)[90, :, :], label='Actual', color='grey', linestyle='-', alpha = 0.4)
    plt.plot(np.array(predictions)[90, :, :], label='Predicted', color='red', linestyle='-', alpha = 0.4)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predictions vs Actual Values')
    plt.legend()
    plt.savefig(cmd + '024_TGCN-G_test_result2/024_TGCN-G_sample' + str(min) + '.jpg', dpi = 600)
    # plt.show()

    a = np.array(actuals)
    p = np.array(predictions)
    np.save(cmd + '024_TGCN-G_test_result2/actual' + str(min) + '.npy', a)
    np.save(cmd + '024_TGCN-G_test_result2/prediction' + str(min) + '.npy', p)


df = pd.DataFrame()
df['MSE'] = min_mse
df['MAE'] = min_mae
df.to_csv(cmd + '024_TGCN-G_test_result2/Accuracy_result.csv', index = False)
