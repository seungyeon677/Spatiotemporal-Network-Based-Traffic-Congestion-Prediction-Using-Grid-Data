import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv

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


class GCLSTM_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(GCLSTM_encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.create_params_layers()
        self.set_params
    
    def create_input_gate(self):
        self.conv_i = ChebConv(in_channels = self.out_channels, out_channels = self.out_channels, K = self.K)
        self.W_i = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_i = nn.Parameter(torch.Tensor(1, self.out_channels))

    def create_forget_gate(self):
        self.conv_f = ChebConv(in_channels = self.out_channels, out_channels = self.out_channels, K = self.K)
        self.W_f = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_f = nn.Parameter(torch.Tensor(1, self.out_channels))

    def create_cell_state(self):
        self.conv_c = ChebConv(in_channels = self.out_channels, out_channels = self.out_channels, K = self.K)
        self.W_c = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_c = nn.Parameter(torch.Tensor(1, self.out_channels))

    def create_output_gate(self):
        self.conv_o = ChebConv(in_channels = self.out_channels, out_channels = self.out_channels, K = self.K)
        self.W_o = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_o = nn.Parameter(torch.Tensor(1, self.out_channels))
    
    def create_params_layers(self):
        self.create_input_gate()
        self.create_forget_gate()
        self.create_cell_state()
        self.create_output_gate()

    def set_params(self):
        glorot(self.W_i)
        glorot(self.W_f)
        glorot(self.W_c)
        glorot(self.W_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def cal_input_gate(self, X, edge_index, edge_weight, H, C):
        I = torch.matmul(X, self.W_i)
        I = I + self.conv_i(H, edge_index, edge_weight)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def cal_forget_gate(self, X, edge_index, edge_weight, H, C):
        F = torch.matmul(X, self.W_f)
        F = F + self.conv_f(H, edge_index, edge_weight)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def cal_cell_state(self, X, edge_index, edge_weight, H, C, I, F):
        T = torch.matmul(X, self.W_c)
        T = T + self.conv_c(H, edge_index, edge_weight)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def cal_output_gate(self, X, edge_index, edge_weight, H, C):
        O = torch.matmul(X, self.W_o)
        O = O + self.conv_o(H, edge_index, edge_weight)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def cal_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(self, X, edge_index, edge_weight, H, C):
        H = self.set_hidden_state(X, H)
        C = self.set_cell_state(X, C)
        I = self.cal_input_gate(X, edge_index, edge_weight, H, C)
        F = self.cal_forget_gate(X, edge_index, edge_weight, H, C)
        C = self.cal_cell_state(X, edge_index, edge_weight, H, C, I, F)
        O = self.cal_output_gate(X, edge_index, edge_weight, H, C)
        H = self.cal_hidden_state(O, C)

        return H, C
    

class GCLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(GCLSTM, self).__init__()
        self.gclstm_encoder = GCLSTM_encoder(in_channels, out_channels, K)

        # Output layer for 10, 20, 30min prediction
        self.fc = nn.Linear(out_channels, 1)
    
    def forward(self, X, edge_index, edge_weight, H, C):
        h_0, c_0 = self.gclstm_encoder(X, edge_index, edge_weight, H, C)
        h = F.relu(h_0)

        out = self.fc(h)

        return out, h


# find best hyperparameter set
def objective(trial):
    # Hyperparameter search space
    out_channels = trial.suggest_categorical("out_channels", [16, 32, 64, 128, 256])
    K = trial.suggest_int("K", 1, 3)
    epochs = trial.suggest_categorical("epoch", [50, 100])
    
    # Initialize model, optimizer, and loss function
    model = GCLSTM(in_channels, out_channels, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train Model
    model.train()
    learning_loss = []
    for epoch in range(epochs):
        total_cost = 0
        H, C = None, None
        for t, snapshot in enumerate(train):
            optimizer.zero_grad()   # 기울기 초기화
            x = snapshot.x
            y = snapshot.y

            out, h = model(x, edge_index, edge_weight, H, C)
            cost = criterion(out, y)

            total_cost += cost.item()

            cost.backward()   # 역전파
            optimizer.step()   # 최적화

        # Average cost
        avg_cost = total_cost / (t + 1)
        learning_loss.append(avg_cost)
    
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

    # Model Hyperparameter
    # in_channel = 7
    # out_channel = 128
    # K = 3


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    # txt 파일로 저장
    with open(cmd + '023_GCLSTM_test_result2/best_params'+ str(min) + '.txt', "w") as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    model = GCLSTM( 
        in_channels, 
        study.best_params['out_channels'],  
        study.best_params['K']
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Train Model
    model.train()
    learning_loss = []
    for epoch in range(study.best_params['epoch']):
        total_cost = 0
        H, C = None, None
        s_time = time.time()
        for t, snapshot in enumerate(train):
            optimizer.zero_grad()   # 기울기 초기화
            x = snapshot.x
            y = snapshot.y

            out, h = model(x, edge_index, edge_weight, H, C)
            cost = criterion(out, y)

            total_cost += cost.item()

            cost.backward()   # 역전파
            optimizer.step()   # 최적화

        # Average cost
        avg_cost = total_cost / (t + 1)
        learning_loss.append(avg_cost)

        e_time = time.time()

        print(f"Epoch {epoch} train MSE: {cost: .4f}")
        print(f"Elapsed Time: {e_time - s_time: .4f}")



    # Make Train Loss Graph
    plt.figure(figsize=(16,8))
    plt.plot(learning_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(cmd + '023_GCLSTM_test_result2/023_GCLSTM_loss_graph' + str(min) +'.jpg', dpi = 600)
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

            out, h = model(x, edge_index, edge_weight, H, C)

            cost = criterion(out, y)
            mae = mae_criterion(out, y)

            total_cost += cost.item()
            total_mae += mae.item()

            predictions.append(out.detach().cpu().numpy())
            actuals.append(y.cpu().numpy())

        cost = total_cost / (t+1)
        mae = total_mae / (t+1)

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
    plt.savefig(cmd + '023_GCLSTM_test_result2/023_GCLSTM_sample' + str(min) + '.jpg', dpi = 600)
    # plt.show()

    a = np.array(actuals)
    p = np.array(predictions)
    np.save(cmd + '023_GCLSTM_test_result2/actual' + str(min) + '.npy', a)
    np.save(cmd + '023_GCLSTM_test_result2/prediction' + str(min) + '.npy', p)


df = pd.DataFrame()
df['MSE'] = min_mse
df['MAE'] = min_mae
df.to_csv(cmd + '023_GCLSTM_test_result2/Accuracy_result.csv', index = False)