"""
Entrypoint for radiomics feature extraction. Uses as input:

- A folder with the images
- A folder with the masks
- A pattern for the identifier of the images (extracted using regex)
- A pattern for globbing the images (glob + regex)
- A pattern for globbing the masks (glob + regex)
- A path to the radiomics config file
- A path to the output folder
"""

import re
import logging
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Pool
from radiomics.featureextractor import RadiomicsFeatureExtractor
from pathlib import Path

MIN_VOXELS = 10

logger = logging.getLogger("gliomaFeatureExtraction")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler("gliomaFeatureExtraction.log")
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)


def check_image_and_mask(image: sitk.Image, mask: sitk.Image) -> None:
    assert (
        image.GetSize() == mask.GetSize()
    ), "Image and mask sizes do not match"
    assert (
        image.GetSpacing() == mask.GetSpacing()
    ), "Image and mask spacings do not match"
    assert (
        image.GetDirection() == mask.GetDirection()
    ), "Image and mask directions do not match"
    assert (
        image.GetOrigin() == mask.GetOrigin()
    ), "Image and mask origins do not match"


def get_image_dict(
    input_folder: Path, identifier_pattern: str, image_pattern: str
) -> dict[str, list[Path]]:
    id_regex = re.compile(identifier_pattern)
    im_regex = re.compile(image_pattern)
    all_images = input_folder.rglob("*nii.gz")
    relevant_images = {}
    for image in all_images:
        identifier = id_regex.search(str(image)).group()
        if im_regex.search(str(image)):
            if identifier not in relevant_images:
                relevant_images[identifier] = []
            relevant_images[identifier].append(image)
    return relevant_images


def extract_case(
    feature_extractor: RadiomicsFeatureExtractor,
    identifier: str,
    image_path: str,
    mask_paths: list[str],
) -> list[dict]:
    image = sitk.ReadImage(image_path)
    mmf = sitk.MinimumMaximumImageFilter()
    mmf.Execute(image)
    logger.info(
        f"{identifier} - image range: {mmf.GetMinimum()} - {mmf.GetMaximum()}"
    )
    im_min, im_max = mmf.GetMinimum(), mmf.GetMaximum()
    image = (image - im_min) / (im_max - im_min)

    curr_features = []
    for mask_path in mask_paths:
        mask = sitk.ReadImage(mask_path)
        check_image_and_mask(image, mask)
        lsf = sitk.LabelStatisticsImageFilter()
        lsf.Execute(mask, mask)
        labels = lsf.GetLabels()
        for label in labels:
            if label == 0:
                continue
            label_sum = lsf.GetSum(label)
            logger.info(f"{identifier} - label {label} has {label_sum} voxels")
            if label_sum < MIN_VOXELS:
                logger.warning(
                    f"{identifier} - label {label} has only {label_sum} voxels (fewer than {MIN_VOXELS})"
                )
                features = {"error": "not enough voxels"}
            else:
                logger.info(
                    f"{identifier} - calculating features with label {label}"
                )
                try:
                    features = feature_extractor.execute(image, mask, label)
                    features["error"] = None
                except Exception as e:
                    logger.error(
                        f"{identifier} - error calculating features with label {label}: {e}"
                    )
                    features = {"error": str(e)}
            features["identifier"] = identifier
            features["mask_label"] = label
            features["image_path"] = image_path
            features["mask_path"] = mask_path
            features["label_sum"] = label_sum
            curr_features.append(features)
    return curr_features


def extract_folder(
    input_folder: Path,
    mask_folder: Path,
    identifier_pattern: str,
    image_pattern: str,
    mask_pattern: str,
    config_path: Path,
    output_path: Path,
    n_jobs: int,
) -> None:
    relevant_images = get_image_dict(
        input_folder, identifier_pattern, image_pattern
    )

    relevant_masks = get_image_dict(
        mask_folder, identifier_pattern, mask_pattern
    )

    logger.info(f"Found {len(relevant_images)} images")
    logger.info(f"Found {len(relevant_masks)} masks")

    if config_path is not None:
        config_path = str(config_path)
    feature_extractor = RadiomicsFeatureExtractor(config_path)

    all_features = []
    all_args = []
    for identifier, images in relevant_images.items():
        if identifier not in relevant_masks:
            logger.warning(f"No mask found for {identifier}")
            continue

        image_path = images[0]
        mask_paths = relevant_masks[identifier]

        all_args.append((feature_extractor, identifier, image_path, mask_paths))

    with Pool(n_jobs) as pool:
        all_features = pool.starmap(extract_case, all_args)
    features_final = []
    for features in all_features:
        features_final.extend(features)

    df = pd.DataFrame(features_final)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features.")

    parser.add_argument(
        "--input_folder",
        type=Path,
        help="Input folder.",
        required=True,
    )
    parser.add_argument(
        "--mask_folder",
        type=Path,
        help="Mask folder.",
        required=True,
    )
    parser.add_argument(
        "--identifier_pattern",
        type=str,
        help="Identifier pattern.",
        required=True,
    )
    parser.add_argument(
        "--image_pattern",
        type=str,
        help="Image pattern.",
        required=True,
    )
    parser.add_argument(
        "--mask_pattern",
        type=str,
        help="Mask pattern.",
        required=True,
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Path to the radiomics config file.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Output path to CSV.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of jobs.",
        default=8,
    )

    args = parser.parse_args()

    extract_folder(**vars(args))
