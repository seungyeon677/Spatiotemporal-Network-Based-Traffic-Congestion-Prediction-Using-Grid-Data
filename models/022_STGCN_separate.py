import torch
import torch.nn as nn
import torch.nn.functional as F
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
feature_np = np.load(cmd + '018_feature_cg6+cr.npy').reshape(4312, 1, 8308, 7)
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


# Build Model
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, x):   # x: (batch_size, seq, num_nodes, node_features) = (1, 6, 8308, 1)
        x = x.permute(0, 3, 2, 1)   # x: (batch_size, node_features, num_nodes, seq)
        p = self.conv1(x)
        q = torch.sigmoid(self.conv2(x))
        pq = p * q
        h = F.relu(pq + self.conv3(x))
        h = h.permute(0, 3, 2, 1)   # h: (batch_size, seq, num_nodes, node_features)

        return h

    
class STConv(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, K):
        super(STConv, self).__init__()
        self._temporal_conv1 = TemporalConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1
            )
        self._graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K
            )
        self._temporal_conv2 = TemporalConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1
            )
        self._batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, x, edge_index, edge_weight):
        T_0 = self._temporal_conv1(x)
        T = torch.zeros_like(T_0).to(device)
        for b in range(T_0.size(0)):
            for t in range(T_0.size(1)):
                T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)

        T = F.relu(T)
        T = self._temporal_conv2(T)
        T = T.permute(0, 2, 1, 3)
        T = self._batch_norm(T)
        T = T.permute(0, 2, 1, 3)

        return T

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, K):
        super(STGCN, self).__init__()
        self.stconv1 = STConv(num_nodes, in_channels, hidden_channels, out_channels, K)
        self.stconv2 = STConv(num_nodes, in_channels, hidden_channels, out_channels, K)

        # Output layer for 10, 20, 30min prediction
        self.fc = nn.Linear(out_channels, 1)
    
    def forward(self, x, edge_index, edge_weight):
        ST_0 = self.stconv1(x, edge_index, edge_weight)
        ST = self.stconv2(x, edge_index, edge_weight)

        output = self.fc(ST)

        return output


# find best hyperparameter set
def objective(trial):
    # Hyperparameter search space
    hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64])
    out_channels = trial.suggest_categorical("out_channels", [64, 128, 256])
    K = trial.suggest_int("K", 1, 3)
    epochs = trial.suggest_categorical("epoch", [50, 100])
    
    # Initialize model, optimizer, and loss function
    model = STGCN(num_nodes, in_channels, hidden_channels, out_channels, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train Model
    model.train()
    learning_loss = []
    for epoch in range(epochs):
        total_cost = 0
        for t, snapshot in enumerate(train_data):
            x = snapshot.x
            y = snapshot.y

            out = model(x, edge_index, edge_weight)
            cost = criterion(out, y)

            total_cost += cost.item()

        # Average cost
        avg_cost = total_cost / (t + 1)
        learning_loss.append(avg_cost)

        optimizer.zero_grad()   # 기울기 초기화
        cost.backward()   # 역전파
        optimizer.step()   # 최적화
    
    return avg_cost


    
num_nodes = 8308
in_channels = 7
lr = 0.001

# min = 10
min_list = [10, 20, 30]
min_mse, min_mae = [], []
for min in min_list:
    target_np = np.load(cmd + '018_target' + str(min) + '_nnor.npy').reshape(4312, 1, 8308, 1)
    target_tensor = torch.tensor(target_np, dtype = torch.float).to(device)

    database = []
    for i in tqdm(range(len(feature_tensor))):
        data = Data(
            x = feature_tensor[i].unsqueeze(0),
            y = target_tensor[i].unsqueeze(0)
        )
        database.append(data)

    # Split Test, Train data
    split_ratio = 0.7
    split_idx = int(len(database) * split_ratio)

    # split_idx = 4172
    train, test = database[:split_idx], database[split_idx:]


    batch_size = 1 
    train_data = DataLoader(train, batch_size=batch_size, shuffle = False) 
    test_data = DataLoader(test, batch_size=batch_size)

    # Model Hyperparameter
    # num_nodes = 8308
    # in_channel = 7
    # hidden_channel = 16
    # out_channel = 128
    # K = 3


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    # txt 파일로 저장
    with open(cmd + '022_STGCN_test_result2/best_params'+ str(min) + '.txt', "w") as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    model = STGCN(
        num_nodes, 
        in_channels, 
        study.best_params['hidden_channels'],
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
        s_time = time.time()
        for t, snapshot in enumerate(train_data):
            x = snapshot.x
            y = snapshot.y

            out = model(x, edge_index, edge_weight)
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
    plt.savefig(cmd + '022_STGCN_test_result2/022_STGCN_loss_graph' + str(min) + '.jpg', dpi = 600)
    # plt.show() 


    # Evaluate Model
    model.eval()
    mae_criterion = nn.L1Loss()
    total_cost = 0
    total_mae = 0
    predictions, actuals = [], []
    with torch.no_grad():
        for t, snapshot in enumerate(tqdm(test_data)):
            x = snapshot.x
            y = snapshot.y

            out = model(x, edge_index, edge_weight)

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
    plt.plot(np.array(actuals)[90, 0, 0, :, :], label='Actual', color='grey', linestyle='-', alpha = 0.4)
    plt.plot(np.array(predictions)[90, 0, 0, :, :], label='Predicted', color='red', linestyle='-', alpha = 0.4)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predictions vs Actual Values')
    plt.legend()
    plt.savefig(cmd + '022_STGCN_test_result2/022_STGCN_sample' + str(min) + '.jpg', dpi = 600)
    # plt.show()

    a = np.array(actuals).reshape(1294, 8308, 1)
    p = np.array(predictions).reshape(1294, 8308, 1)
    np.save(cmd + '022_STGCN_test_result2/actual' + str(min) + '.npy', a)
    np.save(cmd + '022_STGCN_test_result2/prediction' + str(min) + '.npy', p)


df = pd.DataFrame()
df['MSE'] = min_mse
df['MAE'] = min_mae
df.to_csv(cmd + '022_STGCN_test_result2/Accuracy_result.csv', index = False)
