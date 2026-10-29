"""Publish current code plus preserved Python source snapshots; text only."""
import argparse

from . import publish_innovation as extension


def main(message):
    publisher = extension.publisher
    stages = set(publisher.STAGES) | {'2026-10-29_v2_inference_innovation_a'}
    # The old stage collector takes only Markdown/JSON. Include archived source
    # copies explicitly so their source-archive SHA mappings remain reviewable.
    for stage in stages:
        folder = publisher.ROOT / 'archive/research_records' / stage
        for source in folder.rglob('*.py'):
            relative = source.relative_to(publisher.ROOT).as_posix()
            if relative not in publisher.DOCS:
                publisher.DOCS.append(relative)
    extension.main(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--message', required=True)
    main(parser.parse_args().message)
