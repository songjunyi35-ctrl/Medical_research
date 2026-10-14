import argparse
import logging
import datetime
from pathlib import Path
from tqdm import tqdm
from skullstrip import run, configure_device, build_model

def get_logger(log_name: str, logs_dir: str = "./logs"):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        f"{logs_dir}/{log_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

logger = get_logger("SkullStrippingBatch")

if __name__ == "__main__":
    # parse command line
    p = argparse.ArgumentParser(
        description="Robust, universal skull-stripping for brain images of any type."
    )
    p.add_argument(
        "-i",
        "--input_folder",
        metavar="FILE",
        required=True,
        help="input folder with images to skullstrip",
    )
    p.add_argument(
        "-o",
        "--output_folder",
        metavar="FILE",
        required=True,
        help="output folder (will have the same structure as input folder)",
    )
    p.add_argument(
        "-s",
        "--save",
        help="which outputs to save",
        choices=["stripped", "mask", "sdt"],
        default=["mask"],
        nargs="+",
    )
    p.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="use the GPU",
    )
    p.add_argument(
        "-b",
        "--border",
        default=1,
        type=float,
        help="mask border threshold in mm, defaults to 1",
    )
    p.add_argument(
        "-t",
        "--threads",
        type=int,
        help="PyTorch CPU threads, PyTorch default if unset",
    )
    p.add_argument(
        "-f",
        "--fill",
        type=float,
        help="BG fill value, defaults to min(image.min, 0)",
    )
    p.add_argument(
        "--no-csf",
        action="store_true",
        help="exclude CSF from brain border",
    )
    p.add_argument(
        "--model",
        metavar="FILE",
        help="alternative model weights",
    )
    args = p.parse_args()

    device, device_name = configure_device(args.gpu, args.threads)
    model = build_model(device, args.model, args.no_csf)

    path = Path(args.input_folder)
    output_path = Path(args.output_folder)
    if not path.exists():
        raise ValueError(f"Input folder {path} does not exist")

    output_path.mkdir(parents=True, exist_ok=True)

    all_paths = []
    all_paths.extend(list(path.rglob("*nii.gz")))
    all_paths.extend(list(path.rglob("*nii")))
    all_paths.extend(list(path.rglob("*mha")))

    for image_path in tqdm(all_paths):
        out_path_base = str(image_path).replace(str(path), str(output_path))
        out_path_base = out_path_base.rstrip(".nii.gz").rstrip(".nii").rstrip(".mha")

        Path(out_path_base).parent.mkdir(parents=True, exist_ok=True)

        paths = {
            "mask": Path(out_path_base + "_mask.nii.gz"),
            "sdt": Path(out_path_base + "_sdt.nii.gz"),
            "stripped": Path(out_path_base + "_stripped.nii.gz"),
        }

        all_exist = all([paths[k].exists() for k in args.save])

        if all_exist:
            logger.info(f"Skipping {image_path} as it already exists")
            continue

        try:
            run(
                model=model,
                device=device,
                image_path=image_path,
                output_path=paths["stripped"] if "stripped" in args.save else None,
                mask_path=paths["mask"] if "mask" in args.save else None,
                sdt_path=paths["sdt"] if "sdt" in args.save else None,
                fill=args.fill,
                border=args.border,
                logger=logger,
            )
        except ValueError as e:
            continue
