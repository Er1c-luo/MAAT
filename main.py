import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of the GPU to use')
    parser.add_argument('--multi_gpu', type=str2bool, default=True, help='Enable multi-GPU training')
    # Adaptive threshold (used when solver reads these flags; default keeps legacy fixed-percentile path)
    parser.add_argument('--use_adaptive_threshold', type=str2bool, default=False,
                        help='If true, use rolling/window thresholding (requires solver support)')
    parser.add_argument('--threshold_window', type=int, default=200,
                        help='Window length for adaptive threshold statistics')
    parser.add_argument('--threshold_quantile', type=float, default=0.995,
                        help='Quantile for adaptive threshold (e.g. 0.995)')
    # Test-time threshold strategy (solver reads these; fixed = legacy global percentile on combined_energy)
    parser.add_argument('--threshold_mode', type=str, default='fixed',
                        choices=['fixed', 'sliding', 'phase_bucket'],
                        help='fixed | sliding | phase_bucket (phase_bucket wired in solver when supported)')
    parser.add_argument('--phase_period', type=int, default=0,
                        help='Phase cycle length for phase_bucket (0 = use win_size)')
    parser.add_argument('--phase_num_buckets', type=int, default=10,
                        help='Number of phase buckets along one cycle')
    parser.add_argument('--phase_min_count', type=int, default=20,
                        help='Min samples in a phase bucket before bucket-specific quantile; else global fallback')
    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
