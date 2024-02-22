"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class LandlordLstmNewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmNewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class GeneralModel1(nn.Module):
    def __init__(self):
        super().__init__()
        # input: B * 32 * 57
        # self.lstm = nn.LSTM(162, 512, batch_first=True)
        self.conv_z_1 = torch.nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,57)),  # B * 1 * 64 * 32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        # Squeeze(-1) B * 64 * 16
        self.conv_z_2 = torch.nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=(5,), padding=2),  # 128 * 16
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
        )
        self.conv_z_3 = torch.nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=(3,), padding=1), # 256 * 8
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

        )
        self.conv_z_4 = torch.nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=(3,), padding=1), # 512 * 4
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

        )

        self.dense1 = nn.Linear(519 + 1024, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None, debug=False):
        z = z.unsqueeze(1)
        z = self.conv_z_1(z)
        z = z.squeeze(-1)
        z = torch.max_pool1d(z, 2)
        z = self.conv_z_2(z)
        z = torch.max_pool1d(z, 2)
        z = self.conv_z_3(z)
        z = torch.max_pool1d(z, 2)
        z = self.conv_z_4(z)
        z = torch.max_pool1d(z, 2)
        z = z.flatten(1,2)
        x = torch.cat([z,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action, max_value=torch.max(x))


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=(3,),
                               stride=(stride,), padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=(3,),
                               stride=(1,), padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=(1,), stride=(stride,), bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GeneralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 80
        #input 1*54*41
        self.conv1 = nn.Conv1d(40, 80, kernel_size=(3,),
                               stride=(2,), padding=1, bias=False) #1*27*80

        self.bn1 = nn.BatchNorm1d(80)

        self.layer1 = self._make_layer(BasicBlock, 80, 2, stride=2)#1*14*80
        self.layer2 = self._make_layer(BasicBlock, 160, 2, stride=2)#1*7*160
        self.layer3 = self._make_layer(BasicBlock, 320, 2, stride=2)#1*4*320
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(320 * BasicBlock.expansion * 4 + 15 * 4, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, z, x, return_value=False, flags=None, debug=False):
        #print(f"zshape:{z.shape}")
        #print(f"xshape:{x.shape}")
        out = F.relu(self.bn1(self.conv1(z)))
        #print(f"outconv1shape:{out.shape}")
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #print(f"outshape:{out.shape}")
        out = out.flatten(1,2)
        out = torch.cat([x,x,x,x,out], dim=-1)
        #print(f"outflatplusshape:{out.shape}")
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = F.leaky_relu_(self.linear4(out))
        if return_value:
            return dict(values=out)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out,dim=0)[0]
            return dict(action=action, max_value=torch.max(out))


###尝试Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=60):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AttentionFusionLayer(nn.Module):
    def __init__(self, history_dim, combined_dim, fusion_dim):
        super(AttentionFusionLayer, self).__init__()
        self.history_dim = history_dim
        self.combined_dim = combined_dim  # 新的参数，代表场景特征和数值特征合并后的维度
        self.fusion_dim = fusion_dim

        # 对每种特征使用不同的线性层进行维度转换
        self.history_proj = nn.Linear(history_dim, fusion_dim)
        self.combined_proj = nn.Linear(combined_dim, fusion_dim)  # 更新：用于处理合并后的特征

        self.query_layer = nn.Linear(fusion_dim, fusion_dim)
        self.key_layer = nn.Linear(fusion_dim, fusion_dim)
        self.value_layer = nn.Linear(fusion_dim, fusion_dim)

        self.output_layer = nn.Linear(fusion_dim, fusion_dim)

    def forward(self, history_features, combined_features):
        # 首先将不同维度的特征转换到统一的fusion_dim维度
        history_features = self.history_proj(history_features)
        combined_features = self.combined_proj(combined_features)  # 更新：处理合并后的特征

        # 使用history_features作为query，combined_features作为key和value
        query = self.query_layer(history_features)
        key = self.key_layer(combined_features)
        value = self.value_layer(combined_features)

        #print(torch.equal(combined_features[0], combined_features[1])) #false
        #print(torch.equal(value[0], value[1])) #false
        #print(torch.equal(key[0], key[1])) #false
        #print(torch.equal(torch.matmul(query, key.transpose(-2, -1))[0], torch.matmul(query, key.transpose(-2, -1))[1])) #false

        # 计算注意力权重并应用到value上
        attn_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.fusion_dim ** 0.5), dim=-1)
        #print(torch.equal(attn_weights[0], attn_weights[1])) #ture

        attn_output = torch.matmul(attn_weights, value)

        #print(torch.equal(attn_output[0], attn_output[1])) #true

        # 将注意力输出与query（历史特征）进行融合
        combined = attn_output + query

        # 通过输出层
        output = self.output_layer(combined)
        return output


class MultiConv1D(nn.Module):
    def __init__(self):
        super(MultiConv1D, self).__init__()
        # 原始卷积层
        self.conv1 = nn.Conv1d(39, 20, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(20)
        
        # 新增卷积层，具有不同的kernel_size和stride
        self.conv2 = nn.Conv1d(39, 20, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(20)
        
        self.conv3 = nn.Conv1d(39, 20, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(20)
        

    def forward(self, z):
        # 通过各个卷积层
        history_features1 = F.relu(self.bn1(self.conv1(z)))
        history_features2 = F.relu(self.bn2(self.conv2(z)))
        history_features3 = F.relu(self.bn3(self.conv3(z)))
        
        # 合并特征
        combined_features = torch.cat([history_features1, history_features2, history_features3], dim=1)
        
        return combined_features


class GeneralModelTransformer(nn.Module):
    def __init__(self, input_dim=27, d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=256, dropout=0.0, max_seq_length=60):
        super(GeneralModelTransformer, self).__init__()

        self.cnn1d_multi =  MultiConv1D()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder_history = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.transformer_encoder_scene = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.x_dim = 15 + 54 + 15  # Adjust based on your input dimension for x
        self.history_encoder_dim = d_model * 32
        self.scene_encoder_dim = d_model * 7

        # 为地主剩余牌数、地主上家和下家剩余牌数的one-hot编码分别设置嵌入层
        self.landlord_cards_embed = nn.Linear(20, d_model)  # 地主剩余牌数的one-hot编码维度为20
        self.farmer_above_cards_embed = nn.Linear(17, d_model)  # 农民上家剩余牌数的one-hot编码维度为17
        self.farmer_below_cards_embed = nn.Linear(17, d_model)  # 农民下家剩余牌数的one-hot编码维度为17
        
        # 为x设置嵌入层
        self.x_embed = nn.Linear(self.x_dim, d_model)

        # action嵌入层
        self.action_proj = nn.Linear(input_dim, d_model)
        
        
        self.att_fusion_dim = 512  # Dimension for the fused features
        
        self.attention_fusion = AttentionFusionLayer(d_model, d_model * max_seq_length, self.att_fusion_dim)
        self.attention_fusion2 = AttentionFusionLayer(d_model * max_seq_length, d_model, self.att_fusion_dim)

        # 添加批量归一化层
        #self.bn_combined_features = nn.BatchNorm1d(self.history_encoder_dim + self.scene_encoder_dim + numeric_embed_dim * 3 + numeric_embed_dim)  # 新增加的BN层，维度是所有融合特征的维度之和
        #self.bn_combined_features = nn.BatchNorm1d(self.att_fusion_dim + d_model * max_seq_length + d_model)  
        self.bn_combined_features = nn.BatchNorm1d(3924) 
        self.bn_history_features = nn.BatchNorm1d(d_model * max_seq_length)
        self.bn_action_features = nn.BatchNorm1d(d_model)
        #self.linear1 = nn.Linear(self.history_encoder_dim + self.scene_encoder_dim + numeric_embed_dim * 3 + numeric_embed_dim, 1024)
        # 更新linear1的输入维度，因为我们现在使用Attention融合特征
        #self.linear1 = nn.Linear(self.att_fusion_dim + d_model * max_seq_length + d_model, 512)  # 假设融合后的维度为fusion_dim
        self.linear1 = nn.Linear(3924, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, z, x, return_value=False, flags=None, debug=False):
        # z: (action_nums, seq_len, features)
        # Extract features from z
        #print(z.shape)
        history_features = self.cnn1d_multi(z) #(action_nums, 60, 27)
        #print(history_features.shape)

        '''
        history_features = z[:, 2:, :]  # Last 32 seq_len for history  + 2-7 scene_features
        action_features = z[:, 0:1, :]  # seq_len 0 for action features 
        numeric_features = z[:, 1:2, :]  # seq_len 1 for numeric features 

        # 1、嵌入数值特征
        landlord_cards = numeric_features[:, :, :20]
        farmer_above_cards = numeric_features[:, :, 20:37]
        farmer_below_cards = numeric_features[:, :, 37:54]
        
        landlord_cards_embedded = self.landlord_cards_embed(landlord_cards)
        farmer_above_cards_embedded = self.farmer_above_cards_embed(farmer_above_cards)
        farmer_below_cards_embedded = self.farmer_below_cards_embed(farmer_below_cards)
        
        # 2、 嵌入x值
        x_embedded = self.x_embed(x).unsqueeze(1)

        # 3、 嵌入局面特征
        history_features = self.input_proj(history_features)

        
        # 4、 嵌入action特征
        action_features = self.input_proj(action_features)

        # 合并所有局面信息，作为tranformer输入
        history_features = torch.cat([action_features,
                                    landlord_cards_embedded, 
                                    farmer_above_cards_embedded,
                                    farmer_below_cards_embedded,
                                    x_embedded,
                                    history_features], dim=1)

        #print(history_features.shape) #(bz, 43, 64)
        '''
        
        history_features = self.input_proj(history_features)

        history_features = history_features.permute(1, 0, 2)  # Change to (seq_length, batch, features)
        #history_features = self.pos_encoder(history_features) #是否需要pos encoder存疑
        history_features = self.transformer_encoder_history(history_features)
        
        history_features = history_features.permute(1, 2, 0)  

        #transformed_action = action_features[:,0:1].flatten(1)

        history_features = history_features.flatten(1)# Flatten the sequence dimension
        #print(history_features.shape) #(batch, 42 * 64)
        #action_features = action_features.flatten(1)

        #BN
        #history_features = self.bn_history_features(history_features)
        #action_features = self.bn_action_features(action_features)


        #combined_features1 = self.attention_fusion(action_features, history_features)
        #combined_features2 = self.attention_fusion2(history_features, action_features)
        #print(torch.equal(combined_features[0], combined_features[1])) #true

        #combined_features = torch.cat([combined_features1, combined_features2, history_features, action_features], dim=-1)
        #combined_features = torch.cat([combined_features1, history_features, action_features], dim=-1)
        # 在特征融合后立即应用BN层
        
        combined_features = torch.cat([history_features, x], dim=-1)
        combined_features = self.bn_combined_features(combined_features)

        out = self.linear1(combined_features)
        out = F.gelu(out)
        
        out = self.linear2(out)
        out = F.gelu(out)
        
        out = self.linear3(out)
        out = F.gelu(out)
        
        out = F.selu(self.linear4(out))
        
        if return_value:
            return dict(values=out)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out))


'''
class GeneralModelTransformer(nn.Module):
    def __init__(self, input_dim=54, d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=256, dropout=0.0, max_seq_length=32, numeric_embed_dim=16):
        super(GeneralModelTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder_history = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.transformer_encoder_scene = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.x_dim = 15  # Adjust based on your input dimension for x
        self.history_encoder_dim = d_model * 32
        self.scene_encoder_dim = d_model * 7

        # 为地主剩余牌数、地主上家和下家剩余牌数的one-hot编码分别设置嵌入层
        self.landlord_cards_embed = nn.Linear(20, numeric_embed_dim)  # 地主剩余牌数的one-hot编码维度为20
        self.farmer_above_cards_embed = nn.Linear(17, numeric_embed_dim)  # 农民上家剩余牌数的one-hot编码维度为17
        self.farmer_below_cards_embed = nn.Linear(17, numeric_embed_dim)  # 农民下家剩余牌数的one-hot编码维度为17
        
        # 为x设置嵌入层
        self.x_embed = nn.Linear(self.x_dim, numeric_embed_dim)
        
        
        self.att_fusion_dim = 256  # Dimension for the fused features
        
        #self.attention_fusion = AttentionFusionLayer(d_model * 32, d_model * 7, numeric_embed_dim * 3, self.att_fusion_dim)

        # 添加批量归一化层
        self.bn_combined_features = nn.BatchNorm1d(self.history_encoder_dim + self.scene_encoder_dim + numeric_embed_dim * 3 + numeric_embed_dim)  # 新增加的BN层，维度是所有融合特征的维度之和


        self.linear1 = nn.Linear(self.history_encoder_dim + self.scene_encoder_dim + numeric_embed_dim * 3 + numeric_embed_dim, 1024)
        # 更新linear1的输入维度，因为我们现在使用Attention融合特征
        #self.linear1 = nn.Linear(self.att_fusion_dim, 1024)  # 假设融合后的维度为fusion_dim
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, z, x, return_value=False, flags=None, debug=False):
        # z: (action_nums, seq_len, features)
        # Extract features from z
        history_features = z[:, 8:, :]  # Last 32 seq_len for history 
        scene_features = torch.cat([z[:, 0:1, :], z[:, 2:8, :]], dim=1)  # seq_len 0 and 2-7 for scene features 
        numeric_features = z[:, 1:2, :]  # seq_len 2 for numeric features 

        #print(history_features.shape) #(batch, 32, features)
        #print(scene_features.shape) #(batch, 7, features)
        #print(numeric_features.shape) #(batch, 1, features)

        # 1、嵌入数值特征
        landlord_cards = numeric_features[:, :, :20]
        farmer_above_cards = numeric_features[:, :, 20:37]
        farmer_below_cards = numeric_features[:, :, 37:54]
        
        landlord_cards_embedded = self.landlord_cards_embed(landlord_cards.squeeze(1))
        farmer_above_cards_embedded = self.farmer_above_cards_embed(farmer_above_cards.squeeze(1))
        farmer_below_cards_embedded = self.farmer_below_cards_embed(farmer_below_cards.squeeze(1))
        # 将嵌入后的数值特征拼接
        numeric_features_embedded = torch.cat([landlord_cards_embedded, farmer_above_cards_embedded, farmer_below_cards_embedded], dim=-1) 
        #print(numeric_features_embedded.shape) #(batch, 16*3)
        
        # 2、 嵌入x值
        x_embedded = self.x_embed(x)

        # 3、 嵌入时序局面特征
        history_features = history_features.permute(1, 0, 2)  # Change to (seq_length, batch, features)
        history_features = self.input_proj(history_features)
        history_features = self.pos_encoder(history_features)
        history_features = self.transformer_encoder_history(history_features)
        history_features = history_features.permute(1, 2, 0).flatten(1)  # Flatten the sequence dimension
        #print(history_features.shape) #(batch, 2048)


        # 4、嵌入时序无关的局面特征
        scene_features = scene_features.permute(1, 0, 2)  # Change to (seq_length, batch, features)
        scene_features = self.input_proj(scene_features)
        #时序无关的局面特征，不需要pos_encoder编码
        #scene_features = self.pos_encoder(history_features)
        scene_features = self.transformer_encoder_scene(scene_features)
        scene_features = scene_features.permute(1, 2, 0).flatten(1)  # Flatten the sequence dimension
        #print(scene_features.shape) #(batch, 448)


        # 将x特征直接拼接到融合特征中
        combined_features = torch.cat([history_features, scene_features, numeric_features_embedded, x_embedded], dim=-1)
        
        

        #combined_features = self.attention_fusion(history_features, scene_features, numeric_features_embedded)
        
        # 在特征融合后立即应用BN层
        combined_features = self.bn_combined_features(combined_features)

        out = self.linear1(combined_features)
        out = F.gelu(out)
        
        out = self.linear2(out)
        out = F.gelu(out)
        
        out = self.linear3(out)
        out = F.gelu(out)
        
        out = F.gelu(self.linear4(out))
        
        if return_value:
            return dict(values=out)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out))
'''


class BidModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(114, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None, debug=False):
        x = self.dense1(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        x = self.dense2(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        x = self.dense3(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        x = self.dense4(x)
        x = F.leaky_relu(x)
        # x = F.relu(x)
        x = self.dense5(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action, max_value=torch.max(x))


# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['landlord'] = LandlordLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel
model_dict_new = {}
model_dict_new['landlord'] = GeneralModel
model_dict_new['landlord_up'] = GeneralModel
model_dict_new['landlord_down'] = GeneralModel
model_dict_new['bidding'] = BidModel
model_dict_lstm = {}
model_dict_lstm['landlord'] = GeneralModel
model_dict_lstm['landlord_up'] = GeneralModel
model_dict_lstm['landlord_down'] = GeneralModel

model_dict_transformer = {}
model_dict_transformer['landlord'] = GeneralModelTransformer
model_dict_transformer['landlord_up'] = GeneralModelTransformer
model_dict_transformer['landlord_down'] = GeneralModelTransformer
model_dict_new['bidding'] = BidModel

class General_Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        # model = GeneralModel().to(torch.device(device))
        self.models['landlord'] = GeneralModel1().to(torch.device(device))
        self.models['landlord_up'] = GeneralModel1().to(torch.device(device))
        self.models['landlord_down'] = GeneralModel1().to(torch.device(device))
        self.models['bidding'] = BidModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None, debug=False):
        model = self.models[position]
        return model.forward(z, x, training, flags, debug)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['bidding'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()
        self.models['bidding'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models

class OldModel:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['landlord'] = LandlordLstmModel().to(torch.device(device))
        self.models['landlord_up'] = FarmerLstmModel().to(torch.device(device))
        self.models['landlord_down'] = FarmerLstmModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models


class ModelResNet:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        # model = GeneralModel().to(torch.device(device))
        self.models['landlord'] = GeneralModel().to(torch.device(device))
        self.models['landlord_up'] = GeneralModel().to(torch.device(device))
        self.models['landlord_down'] = GeneralModel().to(torch.device(device))
        self.models['bidding'] = BidModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None, debug=False):
        model = self.models[position]
        return model.forward(z, x, training, flags, debug)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['bidding'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()
        self.models['bidding'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models


class ModelTransformer:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        # model = GeneralModel().to(torch.device(device))
        self.models['landlord'] = GeneralModelTransformer().to(torch.device(device))
        self.models['landlord_up'] = GeneralModelTransformer().to(torch.device(device))
        self.models['landlord_down'] = GeneralModelTransformer().to(torch.device(device))
        self.models['bidding'] = BidModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None, debug=False):
        model = self.models[position]
        return model.forward(z, x, training, flags, debug)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['bidding'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()
        self.models['bidding'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
