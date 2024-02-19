import multiprocessing as mp
import pickle
from douzero.env.game import GameEnv
from douzero.dmc.models import Model  # 假设您的模型加载逻辑在这里
import numpy as np
import torch

# 假设DeepAgent是您用来进行游戏评估的Agent
from douzero.evaluation.deep_agent import DeepAgent

def load_card_play_models(card_play_model_path_dict):
    """
    加载斗地主游戏的模型。
    """
    players = {}
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players

def simulate_game(card_play_data, card_play_model_path_dict, data_collection_q):
    """
    模拟单个斗地主游戏，并收集soft labels。
    """
    players = load_card_play_models(card_play_model_path_dict)
    env = GameEnv(players)
    env.card_play_init(card_play_data)

    collected_data = []

    while not env.game_over:
        current_player = env.get_current_player()
        infoset = env.get_infoset()
        
        action, action_probs = current_player.act(infoset, return_action_probs=True)
        
        collected_data.append({
            'position': env.current_pos,
            'game_state': infoset,
            'action_probs': action_probs.cpu().numpy() if torch.is_tensor(action_probs) else action_probs
        })
        
        env.step(action)

    data_collection_q.put(collected_data)

def collect_and_save_data(eval_data, card_play_model_path_dict, num_processes, output_file):
    """
    收集游戏评估过程中生成的soft labels并保存到文件。
    """
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    ctx = mp.get_context('spawn')
    data_collection_q = ctx.SimpleQueue()
    processes = []

    for card_play_data in card_play_data_list:
        p = ctx.Process(target=simulate_game, args=(card_play_data, card_play_model_path_dict, data_collection_q))
        p.start()
        processes.append(p)

    all_collected_data = []
    for _ in range(len(card_play_data_list)):
        all_collected_data.extend(data_collection_q.get())

    with open(output_file, 'wb') as f:
        pickle.dump(all_collected_data, f)

    for p in processes:
        p.join()

if __name__ == "__main__":
    eval_data = '/content/Douzero_Resnet/eval_data.pkl'
    card_play_model_path_dict = {
        'landlord': '/content/Douzero_Resnet/baselines/resnet/resnet_landlord.ckpt',
        'landlord_up': '/content/Douzero_Resnet/baselines/resnet/resnet_landlord_up.ckpt',
        'landlord_down': '/content/Douzero_Resnet/baselines/resnet/resnet_landlord_down.ckpt'
    }
    num_processes = 4
    output_file = 'collected_soft_labels.pkl'
    
    collect_and_save_data(eval_data, card_play_model_path_dict, num_processes, output_file)
