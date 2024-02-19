import os

from douzero.dmc import parser, train
import torch

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    torch.multiprocessing.set_start_method('spawn')
    train(flags)
