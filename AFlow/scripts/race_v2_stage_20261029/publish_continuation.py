"""Include the seed-normalization research source and text in publication."""
import argparse
from . import publish_innovation_v2 as prior


def main(message):
    publisher = prior.extension.publisher
    for stage in ('2026-10-29_race_v2_inference_normalization_a', '2026-10-29_race_v2_inference_soft_context_a'):
        if stage not in publisher.STAGES:
            publisher.STAGES.append(stage)
    for module in ('race_v2_inference_normalization_a_20261029', 'race_v2_inference_soft_context_a_20261029'):
        if module not in publisher.MODULES:
            publisher.MODULES.append(module)
    prior.main(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--message', required=True)
    main(parser.parse_args().message)
