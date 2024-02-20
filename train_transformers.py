import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn.functional as F
import pickle
import numpy as np
from douzero.dmc.models import ModelTransformer,ModelResNet  # 假设你的模型定义在model.py文件中
from douzero.env.env import get_obs
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython.display import clear_output

class DouzeroDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        position = item['position']

        obs = get_obs(item['game_state'], use_general=True)
        z = torch.from_numpy(obs['z_batch']).float()
        x = torch.from_numpy(obs['x_batch']).float()

        action_probs = torch.tensor(item['action_probs'], dtype=torch.float32)
        #print(position)
        #print(f"zshape:{z.shape}")
        #print(f"xshape:{x.shape}")
        #print(f"action_probsshape:{action_probs.shape}")
        
        return position, z, x, action_probs

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (position, z, x, action_probs) in enumerate(tqdm(train_loader)):
        # 使用squeeze方法去除batch_size维度
        z = torch.squeeze(z, 0).to(device)
        x = torch.squeeze(x, 0).to(device)
        action_probs = torch.squeeze(action_probs, 0).to(device)

        optimizer.zero_grad()
        outputs = model.forward(z, x, return_value=True)
        loss = F.mse_loss(outputs['values'], action_probs)
        print("****below outputs****\n")
        print(outputs['values'])
        print("****below action_probs****\n")
        print(action_probs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 1000 == 0:  # 这里的10是一个可调整的值，根据你的需求和数据集大小调整
            print(total_loss/(batch_idx+1))

    return total_loss / len(train_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    with open('/content/Douzero_Resnet/collected_soft_labels.pkl', 'rb') as f:
        data = pickle.load(f)

    model_wrapper = ModelResNet(device='0' if str(device)=='cuda' else 'cpu')
    optimizers = {
        'landlord': Adam(model_wrapper.get_model('landlord').parameters(), lr=1e-3),
        'landlord_up': Adam(model_wrapper.get_model('landlord_up').parameters(), lr=1e-2),
        'landlord_down': Adam(model_wrapper.get_model('landlord_down').parameters(), lr=1e-2)
    }

    epochs = 10
    for epoch in range(epochs):
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            # 根据position过滤数据
            filtered_data = [item for item in data if item['position'] == position]
            position_dataset = DouzeroDataset(filtered_data)
            position_loader = DataLoader(position_dataset, batch_size=1, shuffle=True)

            print(f"Training {position} model, Epoch {epoch+1}/{epochs}")
            total_loss = train(model_wrapper.get_model(position), position_loader, optimizers[position], device)
            print(f"Loss: {total_loss:.4f}")
            # 保存模型
            torch.save(model_wrapper.get_model(position).state_dict(), f'{position}_model.ckpt')

if __name__ == '__main__':
    main()
