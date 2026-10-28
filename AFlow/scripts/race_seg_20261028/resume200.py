"""User-authorized 200-epoch continuation of the unchanged numerical runner."""
import argparse
import torch
from scripts.race_seg_20261028 import train as original
from scripts.race_seg_20261028.common import R, read, verify_sources


def main(arm):
    verify_sources('RESUME200_FREEZE.json')
    budget = read(R/'BUDGET_DECISION.json')
    assert budget['epochs'] == 200 and budget['microbatch'] == 2
    threads = budget['cpu_threads']
    setup_original = original.setup

    def setup_same_model(selected_arm):
        model = setup_original(selected_arm)
        torch.set_num_threads(threads)
        return model

    original.setup = setup_same_model
    # The frozen runner restores model, optimizer, counters and the original
    # 200-epoch LR schedule. No new initialization is used for training.
    original.train(arm, resume=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=('control', 'race'))
    args = parser.parse_args()
    main(args.arm)
