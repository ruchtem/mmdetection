import numpy as np
from pycocotools.coco import COCO

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class MovingMnistDataset(CocoDataset):

    CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            valid_inds.append(i)
        return valid_inds