"""CLI for training"""

import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import wandb
from cells_instance_segmentation.main_utils import load_config
from cells_instance_segmentation.modules.detectron.trainer import Trainer

if __name__ == '__main__':
    config_path = 'configs/data_config.yaml'
    NUM_EPOCHS = 50
    LR = 0.0005
    BATCH_SIZE = 3

    wandb.init(
        sync_tensorboard=True,
        settings=wandb.Settings(start_method="thread", console="off"),
    )

    config = load_config(path=config_path)
    detectron_config = get_cfg()

    all_annotations_path = config['data']['all_annotations_path']
    train_annotations_path = config['data']['train_annotations_path']
    val_annotations_path = config['data']['val_annotations_path']
    root_data_path = config['data']['root_data_path']

    detectron_config.INPUT.MASK_FORMAT = 'bitmask'
    register_coco_instances(
        name='sartorius_train',
        metadata={},
        json_file=train_annotations_path,
        image_root=root_data_path,
    )
    register_coco_instances(
        name='sartorius_val',
        metadata={},
        json_file=val_annotations_path,
        image_root=root_data_path,
    )
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')

    detectron_config.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    detectron_config.DATASETS.TRAIN = ("sartorius_train",)
    detectron_config.DATASETS.TEST = ("sartorius_val",)
    detectron_config.DATALOADER.NUM_WORKERS = 2
    detectron_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        config_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    detectron_config.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    detectron_config.SOLVER.BASE_LR = LR
    detectron_config.SOLVER.MAX_ITER = (
        len(DatasetCatalog.get(name='sartorius_train'))
        * NUM_EPOCHS
        // detectron_config.SOLVER.IMS_PER_BATCH
    )
    detectron_config.SOLVER.STEPS = []
    detectron_config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    detectron_config.MODEL.ROI_HEADS.NUM_CLASSES = 2
    detectron_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    detectron_config.TEST.EVAL_PERIOD = (
        len(DatasetCatalog.get(name='sartorius_train'))
        // detectron_config.SOLVER.IMS_PER_BATCH
    )  # Once per epoch

    os.makedirs(detectron_config.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(detectron_config)
    trainer.resume_or_load(resume=True)
    trainer.train()
