import multiprocessing as mp
import pickle
from douzero.env.game import GameEnv
from douzero.dmc.models import ModelResNet  # 假设您的模型加载逻辑在这里


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

def simulate_game(card_play_data_list, card_play_model_path_dict, data_collection_q):
    """
    模拟单个斗地主游戏，并收集soft labels。
    """
    players = load_card_play_models(card_play_model_path_dict)
    EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}
    env = GameEnv(players)
    collected_data = []

    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)

        print("\nStart ------- ")
        print ("".join([EnvCard2RealCard[c] for c in card_play_data["landlord"]]))
        print ("".join([EnvCard2RealCard[c] for c in card_play_data["landlord_down"]]))
        print ("".join([EnvCard2RealCard[c] for c in card_play_data["landlord_up"]]))
        
        count = 0
        while not env.game_over:
            
            infoset = env.get_infoset()

            try:
                if len(infoset.legal_actions) > 1:
                    _, action_probs = players[env.acting_player_position].act(env.game_infoset, return_action_probs=True)
                    
                    collected_data.append({
                    'position': env.acting_player_position,
                    'game_state': infoset,
                    'action_probs': action_probs.cpu().numpy() if torch.is_tensor(action_probs) else action_probs
            })
            except:
                print(type(players[env.acting_player_position].act(env.game_infoset, return_action_probs=True)))
            
            
            action = env.step()

            '''
            if count % 3 == 2:
                    end = "\n"
            else:
                end = "   "
            if len(action) == 0:
                print("Pass", end=end)
            else:
                print("".join([EnvCard2RealCard[c] for c in action]), end=end)
            count+=1
            '''
        
        env.reset()

    data_collection_q.put(collected_data)


def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker


def collect_and_save_data(eval_data, card_play_model_path_dict, num_processes, output_file):
    """
    收集游戏评估过程中生成的soft labels并保存到文件。
    """
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)
    print("eval_data load success.")

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_processes)
    del card_play_data_list

    ctx = mp.get_context('spawn')
    data_collection_q = ctx.SimpleQueue()
    processes = []

    for card_play_data in card_play_data_list_each_worker:
        p = ctx.Process(target=simulate_game, args=(card_play_data, card_play_model_path_dict, data_collection_q))
        p.start()
        processes.append(p)

    all_collected_data = []
    for _ in range(num_processes):
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
    num_processes = 8
    output_file = 'collected_soft_labels.pkl'
    
    collect_and_save_data(eval_data, card_play_model_path_dict, num_processes, output_file)
