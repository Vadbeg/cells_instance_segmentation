"""CLI for showing data with annotations"""

import random
from pathlib import Path

import typer
import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from cells_instance_segmentation.main_utils import load_config


def _create_coco(annotation_file_path: Path) -> COCO:
    coco = COCO(annotation_file=annotation_file_path)
    
    return coco


def _show_images(coco: COCO, images_root: Path, num_of_images: int) -> None:
    image_ids = coco.getImgIds()
    image_anns = coco.loadImgs(ids=random.choices(image_ids, k=num_of_images))

    _, axs = plt.subplots(
        nrows=len(image_anns),
        ncols=2,
        figsize=(20, 7 * len(image_anns))
    )
    
    for curr_image_ann, curr_ax in zip(image_anns, axs):
        filepath = images_root.joinpath(curr_image_ann['file_name'])
        image = io.imread(str(filepath))
        
        annotation_ids = coco.getAnnIds(imgIds=[curr_image_ann['id']])
        annotations = coco.loadAnns(ids=annotation_ids)
        
        curr_ax[0].imshow(image)
        curr_ax[1].imshow(image)

        plt.sca(curr_ax[1])
        coco.showAnns(annotations, draw_bbox=True)
        
    plt.show()


def show_data(
    config_path: Path = typer.Option(
        ..., help='Path to config with paths'
    ),
    num_of_images: int = typer.Option(
        default=3, help='Number of images to show'
    )
) -> None:
    config = load_config(path=config_path)

    all_annotations_file = Path(config['data']['all_annotations_file'])
    root_data_path = Path(config['data']['root_data_path'])
    
    coco = _create_coco(annotation_file_path=all_annotations_file)
    _show_images(
        coco=coco,
        images_root=root_data_path,
        num_of_images=num_of_images
    )
