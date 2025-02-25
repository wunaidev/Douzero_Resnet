import os
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time
from douzero.radam.radam import RAdam

import torch
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return Env(flags.objective)

def get_batch(b_queues, position, flags, lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    b_queue = b_queues[position]
    buffer = []
    while len(buffer) < flags.batch_size:
        buffer.append(b_queue.get())
    batch = {
        key: torch.stack([m[key] for m in buffer], dim=1)
        for key in ["done", "episode_return", "target", "obs_z", "obs_x_batch", "obs_type"]
    }
    del buffer
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down', 'bidding']
    optimizers = {}
    for position in positions:
        optimizer = RAdam(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            eps=flags.epsilon)
        optimizers[position] = optimizer
    return optimizers


def act(i, device, batch_queues, model, flags):
    positions = ['landlord', 'landlord_up', 'landlord_down', 'bidding']
    for pos in positions:
        model.models[pos].to(torch.device(device if device == "cpu" else ("cuda:"+str(device))))
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}
        type_buf = {p: [] for p in positions}
        obs_x_batch_buf = {p: [] for p in positions}

        position_index = {"landlord": 31, "landlord_up": 32, "landlord_down": 33}
        bid_type_index = {"landlord": 41, "landlord_up": 42, "landlord_down": 43}
        bid_type_map = {41: "landlord", 42: "landlord_up", 43: "landlord_down"}

        position, obs, env_output = env.initial(model, device, flags=flags)
        bid_obs_buffer = env_output["begin_buf"]["bid_obs_buffer"]
        multiply_obs_buffer = env_output["begin_buf"]["multiply_obs_buffer"]
        while True:
            # print("posi", position)
            for bid_obs in bid_obs_buffer:
                obs_z_buf["bidding"].append(bid_obs['z_batch'])
                obs_x_batch_buf["bidding"].append(bid_obs["x_batch"])
                type_buf["bidding"].append(bid_type_index[bid_obs["position"]])
                size["bidding"] += 1
            for mul_obs in multiply_obs_buffer:
                obs_z_buf[mul_obs["position"]].append(mul_obs['z_batch'])
                obs_x_batch_buf[mul_obs["position"]].append(mul_obs["x_batch"])
                type_buf[mul_obs["position"]].append(2)
                size[mul_obs["position"]] += 1
            while True:

                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                #print(f"obs['z_batch']:{obs['z_batch'].shape}")
                #print(f"obs['x_batch']:{obs['x_batch'].shape}")
                #print(type(agent_output))
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]

                #print(type(env_output))
                #print(env_output.keys())

                '''
                print(torch.vstack((_cards2tensor(action).unsqueeze(0), env_output['obs_z'])).float().shape)
                print(env_output['obs_x_no_action'].float().shape)
                obs_z_buf[position].append(torch.vstack((_cards2tensor(action).unsqueeze(0), env_output['obs_z'])).float())
                # x_batch = torch.cat((env_output['obs_x_no_action'], _cards2tensor(action)), dim=0).float()
                x_batch = env_output['obs_x_no_action'].float()
                obs_x_batch_buf[position].append(x_batch)
                '''
                
                #print(obs['z_batch'][_action_idx].shape)
                #print(obs['x_batch'][_action_idx].shape)
                obs_z_buf[position].append(obs['z_batch'][_action_idx])
                obs_x_batch_buf[position].append(obs['x_batch'][_action_idx])

                type_buf[position].append(position_index[position])
                position, obs, env_output = env.step(action, model, device, flags=flags)
                size[position] += 1
                if env_output['done']:
                    bid_obs_buffer = env_output["begin_buf"]["bid_obs_buffer"]
                    multiply_obs_buffer = env_output["begin_buf"]["multiply_obs_buffer"]
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        # print(p, diff)
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)
                            if p != "bidding":
                                episode_return = env_output['episode_return']["play"][p] if p == 'landlord' else -env_output['episode_return']["play"][p]
                                episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                                episode_return_buf[p].append(episode_return)
                                target_buf[p].extend([episode_return for _ in range(diff)])
                            else:
                                offset = len(target_buf[p])
                                for index in range(diff):
                                    pos = type_buf[p][index+offset]
                                    if pos == 41:
                                        episode_return = env_output['episode_return']["bid"]["landlord"]
                                    else:
                                        episode_return = -env_output['episode_return']["bid"][bid_type_map[pos]]
                                    episode_return_buf[p].append(episode_return)
                                    # print(p, episode_return)
                                    target_buf[p].append(episode_return)
                    break
            for p in positions:
                if size[p] > T:
                    # print(p, "epr", torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in episode_return_buf[p][:T]]),)
                    batch_queues[p].put({
                        "done": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in done_buf[p][:T]]),
                        "episode_return": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in episode_return_buf[p][:T]]),
                        "target": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in target_buf[p][:T]]),
                        "obs_z": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in obs_z_buf[p][:T]]),
                        "obs_x_batch": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in obs_x_batch_buf[p][:T]]),
                        "obs_type": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in type_buf[p][:T]])
                    })
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_batch_buf[p] = obs_x_batch_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    type_buf[p] = type_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    if len(list_cards) == 0:
        return torch.zeros(54, dtype=torch.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    matrix = np.concatenate((matrix.flatten('F'), jokers))
    matrix = torch.from_numpy(matrix)
    return matrix
