"""Publish recovery code, original frozen choices and text evidence only."""
import argparse
from scripts.race_v2_stage_20261029 import publish_continuation as base


def main(message):
    publisher = base.prior.extension.publisher
    for stage in ('2026-10-29_race_v2_final270_recovery',):
        if stage not in publisher.STAGES:
            publisher.STAGES.append(stage)
    module = 'race_v2_final270_recovery_20261029'
    if module not in publisher.MODULES:
        publisher.MODULES.append(module)
    base.main(message)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--message',required=True)
    main(parser.parse_args().message)
