"""Publish the inference innovation extension with the existing text-only rules."""
import argparse
import subprocess

from . import publish_snapshot_v2 as publisher


def git(*arguments, env=None):
    # Porcelain has meaningful leading status spaces; preserve them on retry.
    return subprocess.check_output(['git', *arguments], cwd=publisher.RELEASE,
                                   env=env, text=True).rstrip('\n')


def main(message):
    publisher.git = git
    stage = '2026-10-29_v2_inference_innovation_a'
    module = 'race_v2_inference_innovation_a_20261029'
    if stage not in publisher.STAGES:
        publisher.STAGES.append(stage)
    if module not in publisher.MODULES:
        publisher.MODULES.append(module)
    publisher.main(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--message', required=True)
    main(parser.parse_args().message)
