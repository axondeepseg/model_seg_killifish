"""
Prepares a new dataset for nnUNetv2 for 3-class segmentation: myelinated axons 
(axon + myelin) and unmyelinated axons.

This script reads a JSON file associating original filenames to nnUNet case IDs,
and adds anything that is not in the JSON file to the dataset as a new case.
It also processes the masks to ensure they are in the correct format for nnUNet.
"""

MASK_SUFFIXES_TO_LOAD = [
    "_seg-uaxon-manual.png", 
    "_seg-axon-manual.png", 
    "_seg-myelin-manual.png"
]
CLASS_MAPPING = {
    "background": 0,
    "uaxon": 1,
    "myelin": 2,
    "axon": 3
}
DATASETNAME = "TEM_KILLIFISH"
DATASETDESC = "TEM dataset of killifish myelinated and unmyelinated axons."
DATASETID = '401'

__author__ = "Armand Collin"
__license__ = "MIT"

import argparse
import json
import cv2
import os
import numpy as np
from typing import List, Dict
from pathlib import Path


def read_masks(file_basename: Path) -> np.ndarray:
    """
    Reads the masks for a given file basename and returns a combined mask.
    The masks are expected to be in the same directory as the image, with specific suffixes.
    """
    combined_mask = None
    for classname, class_id in CLASS_MAPPING.items():
        if classname == "background":
            continue

        mask_path = file_basename.with_name(f"{file_basename.stem}_seg-{classname}-manual.png")
        if not mask_path.exists():
            # unmyelinated axon mask is not always present
            if classname == "uaxon":
                continue
            raise FileNotFoundError(f"Mask file {mask_path} does not exist.")
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if combined_mask is None:
            combined_mask = np.zeros(mask.shape, dtype=np.uint8)
        combined_mask[mask > 0] = class_id
    
    return combined_mask

def create_dirs(base_dir: Path, subdirs: List[str]):
    for subdir in subdirs:
        dir = str(base_dir / subdir)
        os.makedirs(dir, exist_ok=True)

def main(datapath: Path, json_path: Path, output_dir: Path):
    
    # setup the nnunet dataset structure
    dataset_name = f"Dataset{DATASETID}_{DATASETNAME}"
    out_folder = output_dir / 'nnUNet_raw' / dataset_name
    create_dirs(out_folder, ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs'])
    image_folder = out_folder / 'imagesTr'
    label_folder = out_folder / 'labelsTr'

    # load the JSON file if it exists
    if json_path is not None:
        with open(json_path, 'r') as f:
            fname_data = json.load(f)
            existing_fnames = set(fname_data.keys())
            max_id = max(fname_data.values(), default=0)
    else:
        fname_data = {}
        existing_fnames = set()
        max_id = 0

    # index new images
    all_pngs = list(datapath.glob('*.png'))
    all_images = [img for img in all_pngs if not any(suffix in img.name for suffix in MASK_SUFFIXES_TO_LOAD)]
    all_images = [img for img in all_images if not "_seg-axonmyelin-manual" in img.name]
    new_images = [img for img in all_images if img.name not in existing_fnames]
    print(f"Found {len(all_images)} images, {len(new_images)} new images to add to the dataset.")

    # process new images and associated masks
    current_case_id = max_id + 1
    for impath in new_images:
        nnunet_mask = read_masks(impath)
        # copy the image to the nnUNet imagesTr folder
        nnunet_image_path = image_folder / f"{DATASETNAME}_{current_case_id:03d}_0000.png"
        image = cv2.imread(str(impath), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(nnunet_image_path), image)

        # save the mask
        nnunet_mask_path = label_folder / f"{DATASETNAME}_{current_case_id:03d}.png"
        cv2.imwrite(str(nnunet_mask_path), nnunet_mask)

        # update fname_data
        fname_data[impath.name] = current_case_id
        current_case_id += 1
    
    # update json filename-case_id mapping
    if json_path is not None:
        with open(json_path, 'w') as f:
            json.dump(fname_data, f, indent=2)
    else:
        # create a new JSON file if it does not exist
        with open(output_dir / 'filename_case_id_mapping.json', 'w') as f:
            json.dump(fname_data, f, indent=2)

    dataset_info = {
        "name": DATASETNAME,
        "description": DATASETDESC,
        "labels": CLASS_MAPPING,
        "channel_names": {"0": "rescale_to_0_1"},
        "numTraining": len(fname_data),
        "numTest": 0,
        "file_ending": ".png",
    }
    with open(out_folder / 'dataset.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "DATAPATH",
        type=Path,
        help="Path to the dataset directory containing images and masks to add " \
        "to the nnUNet dataset."
    )
    ap.add_argument(
        "-j",
        type=Path,
        required=False,
        default=None,
        help="Path to the JSON file associating original filenames to nnUNet case IDs."
    )
    ap.add_argument(
        "-o",
        type=Path,
        required=False,
        default=None,
        help="Path to the output directory where the nnUNet dataset is. If unspecified, " \
        "the script will create a new one." 
    )
    args = ap.parse_args()
    if not args.DATAPATH.is_dir():
        raise ValueError(f"DATAPATH {args.DATAPATH} is not a directory.")
    if args.j is not None and not args.j.is_file():
        raise ValueError(f"JSON file {args.j} does not exist.")
    if args.o is not None and not args.o.exists():
        raise ValueError(f"Output directory {args.o} does not exist.")
    
    if args.o is None:
        args.o = Path(f'NNUNET_{DATASETNAME}')
        # create the output directory if it does not exist
        args.o.mkdir(exist_ok=True)
    else:
        if not (args.o / 'nnUNet_raw' /f"Dataset{DATASETID}_{DATASETNAME}").exists():
            raise ValueError(f"Output directory {args.o} does not contain the expected nnUNet dataset structure.")
    
    main(args.DATAPATH, args.j, args.o)