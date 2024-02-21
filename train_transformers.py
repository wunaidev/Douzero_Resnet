import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch.nn.functional as F
import pickle
import numpy as np
from douzero.dmc.models import ModelTransformer,ModelResNet  # 假设你的模型定义在model.py文件中
from douzero.env.env import get_obs
from tqdm import tqdm
from douzero.evaluation.simulation import evaluate
from generate_eval_data import generate
from collect_soft_labels import collect_and_save_data

class DouzeroDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        position = item['position']

        obs = _get_obs_transformer(item['game_state'], use_general=True)
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
    correct_predictions = 0
    total_predictions = 0
    for batch_idx, (position, z, x, action_probs) in enumerate(tqdm(train_loader)):
        z = torch.squeeze(z, 0).to(device)
        x = torch.squeeze(x, 0).to(device)
        action_probs = torch.squeeze(action_probs, 0).to(device)

        optimizer.zero_grad()
        outputs = model.forward(z, x, return_value=True)
        
        # 调整outputs['values']的形状以适应F.cross_entropy
        # 假设outputs['values']的形状原本为[action_nums, 1]，我们需要将其变为[1, action_nums]来模拟batch_size为1的情况
        q_values = outputs['values'].squeeze(-1)  # 将形状从[action_nums, 1]变为[action_nums]
        t_value = action_probs.squeeze(-1)
        # 计算MSE损失
        #mse_loss = F.mse_loss(q_values, t_value)
        
        # 为了使用F.cross_entropy，我们需要确保q_values是[batch_size, num_classes]的形状
        # 在这个案例中，batch_size=1，所以我们需要添加一个维度来模拟它
        q_values = q_values.unsqueeze(0)  # 将形状从[action_nums]变为[1, action_nums]
        t_value = t_value.unsqueeze(0)  # 将形状从[action_nums]变为[1, action_nums]

        # 计算交叉熵损失
        max_indices = t_value.argmax(dim=1)
        ce_loss = F.cross_entropy(q_values, max_indices.long())  # 确保max_indices是长整型，并且有一个额外的维度
  

        # 计算KLDivLoss
        criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
        log_probs = F.log_softmax(q_values, dim=1)
        kl_loss = criterion_kl(log_probs, t_value)


        # 组合两部分的损失
        #loss = kl_loss + ce_loss
        loss =  ce_loss



        if batch_idx % 1501 == 0:
            #print("****below outputs****\n")
            #print(outputs['values'])
            #print(torch.argmax(outputs['values'], dim=0)[0])
            #print("****below action_probs****\n")
            #print(action_probs)
            #print(torch.argmax(action_probs, dim=0)[0])
            pass

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 计算准确率
        predicted = q_values.argmax(dim=1)  # 预测的最佳动作索引
        correct_predictions += (predicted == max_indices).sum().item()
        total_predictions += max_indices.size(0)

        if batch_idx % 1000 == 0:
            print(f"Average Loss: {total_loss/(batch_idx+1):.4f}")

    accuracy = 100. * correct_predictions / total_predictions
    print(f"Training Accuracy: {accuracy:.4f}%")
    return total_loss / len(train_loader)


def print_grad_hook(grad):
    print(grad)

def register_gradient_hooks(model):
    for name, parameter in model.named_parameters():
        parameter.register_hook(print_grad_hook)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    with open('/content/Douzero_Resnet/collected_soft_labels.pkl', 'rb') as f:
        data = pickle.load(f)

    if which_model == "resnet":
        model_wrapper = ModelResNet(device='0' if str(device) == 'cuda' else 'cpu')
    else:
        model_wrapper = ModelTransformer(device='0' if str(device) == 'cuda' else 'cpu')
    print(f"正在训练:{which_model}")
    
    # 加载模型权重（如果存在）
    weight_files = {
        'landlord': f'{which_model}_landlord_model.ckpt',
        'landlord_up': f'{which_model}_landlord_up_model.ckpt',
        'landlord_down': f'{which_model}_landlord_down_model.ckpt'
    }
    for position, weight_file in weight_files.items():
        try:
            model_wrapper.get_model(position).load_state_dict(torch.load(weight_file))
            print(f"Successfully loaded weights from {weight_file} for {position}")
        except FileNotFoundError:
            print(f"No weight file found for {position} at {weight_file}, initializing from scratch.")
    
    optimizers = {
        'landlord': AdamW(model_wrapper.get_model('landlord').parameters(), lr=1e-4),
        'landlord_up': AdamW(model_wrapper.get_model('landlord_up').parameters(), lr=1e-4),
        'landlord_down': AdamW(model_wrapper.get_model('landlord_down').parameters(), lr=1e-4)
    }

    epochs = 10000
    for epoch in range(epochs):
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            #register_gradient_hooks(model_wrapper.get_model(position))
            #model_wrapper.get_model(position).linear3.weight.register_hook(lambda grad: print(grad))
            # 根据position过滤数据
            filtered_data = [item for item in data if item['position'] == position]
            position_dataset = DouzeroDataset(filtered_data)
            position_loader = DataLoader(position_dataset, batch_size=1, shuffle=True)

            print(f"Training {position} model, Epoch {epoch+1}/{epochs}")
            print(f"训练数据:{len(position_loader)}条")
            total_loss = train(model_wrapper.get_model(position), position_loader, optimizers[position], device)
            print(f"Loss: {total_loss:.4f}")
            # 保存模型
            torch.save(model_wrapper.get_model(position).state_dict(), f'{which_model}_{position}_model.ckpt')

        evaluate(f"{which_model}_landlord_model.ckpt",
            f"/content/Douzero_Resnet/baselines/sl/landlord_up.ckpt",
            f"/content/Douzero_Resnet/baselines/sl/landlord_down.ckpt",
            f"eval_data.pkl",
            8,
            False,
            False,
            "NEW")

        evaluate(f"/content/Douzero_Resnet/baselines/sl/landlord.ckpt",
            f"{which_model}_landlord_up_model.ckpt",
            f"{which_model}_landlord_down_model.ckpt",
            f"eval_data.pkl",
            8,
            False,
            False,
            "NEW")


        if (epoch+1)%10==0:
            print("正在生成新的数据...")
            eval_data = '/content/Douzero_Resnet/eval_data.pkl'

            print("output_pickle:", eval_data)

            data = []
            for _ in range(1000):
                data.append(generate())

            print("saving pickle file...")
            with open(eval_data,'wb') as g:
                pickle.dump(data,g,pickle.HIGHEST_PROTOCOL)

            
            card_play_model_path_dict = {
                'landlord': '/content/Douzero_Resnet/baselines/resnet/resnet_landlord.ckpt',
                'landlord_up': '/content/Douzero_Resnet/baselines/resnet/resnet_landlord_up.ckpt',
                'landlord_down': '/content/Douzero_Resnet/baselines/resnet/resnet_landlord_down.ckpt'
            }
            num_processes = 8
            output_file = 'collected_soft_labels.pkl'
            
            collect_and_save_data(eval_data, card_play_model_path_dict, num_processes, output_file)
            del data
            with open('/content/Douzero_Resnet/collected_soft_labels.pkl', 'rb') as f:
                data = pickle.load(f)
            
        
if __name__ == '__main__':
    #which_model = "resnet"
    which_model = "transformer"
    main()