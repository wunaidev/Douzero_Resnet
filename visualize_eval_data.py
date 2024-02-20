import pickle
from douzero.env.env import get_obs

def load_collected_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def display_sample_data_with_last_move(collected_data, num_samples=5):
    print(f"展示每个位置的前{num_samples}个样本数据...\n")
    positions = ['landlord', 'landlord_up', 'landlord_down']
    samples_shown = {position: 0 for position in positions}
    
    for data in collected_data:
        position = data['position']
        if samples_shown[position] < num_samples:
            game_state = data['game_state']
            obs = get_obs(data['game_state'])
            print(f"位置: {position}")
            print(f"手牌: {game_state.player_hand_cards}")
            print(f"法定动作: {game_state.legal_actions}")
            print(f"动作概率分布: {data['action_probs']}")
            action_idx = data['action_probs'].argmax()
            print(f"最可能的动作: {data['game_state'].legal_actions[action_idx]}") if len(data['game_state'].legal_actions) > action_idx else print("最可能的动作: N/A")


            last_action = game_state.last_move
            last_pid = game_state.last_pid
            print(f"上一轮出牌的玩家: {last_pid}")
            if last_action is not None and len(last_action) > 0:
                print(f"上一轮出的牌: {''.join([str(card) for card in last_action])}")
            else:
                print("上一轮出的牌: 无（可能是过牌）")
            
            print(f"当前视角信息（部分）: x_batch: {obs['x_batch'][0]}, z_batch: {obs['z_batch'][0].shape}")
            print("------\n")
            samples_shown[position] += 1
        if all(count >= num_samples for count in samples_shown.values()):
            break

            
if __name__ == "__main__":
    file_path = 'collected_soft_labels.pkl'  # 更新为您的文件路径
    collected_data = load_collected_data(file_path)
    display_sample_data_with_last_move(collected_data, num_samples=5)  # 可以调整num_samples来展示更多或更少的样本
