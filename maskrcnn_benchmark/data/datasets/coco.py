# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

import cv2
import numpy as np
import pycocotools.mask as maskUtils

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

def poly2rle(segm, w, h):
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    return rle

def generate_pyramid_label(H, W, corner_points):
    """

    :param int H: image_H
    :param int W: image_W
    :param np.ndarray corner_points: dtype=np.float32, shape=[point_num, {x,y}] 3 <= point_num <= 8
    :return: np.ndarray ans: dtype=np.float32, shape=[H, W]

    generate a pyramid mask from corner_points 
      within the bounding box {box_top=0, box_bottom=H, box_left=0, box_right=W}
    """
    center = corner_points.mean(axis=0)
    vectors = corner_points - center
    matrices = np.empty((4, 2, 2), dtype=np.float32)
    for i in range(4):
        m = vectors[[i, (i + 1) % 4]].T
        matrices[i] = np.linalg.pinv(m)
    points = np.empty((H, W, 2), dtype=np.float32)  # H, W, {x, y}
    points[:, :, 0] = np.arange(W)
    points[:, :, 1] = np.arange(H)[..., None]
    points -= center
    ans: np.ndarray = np.matmul(matrices[:, None, None, ...], points[..., None])
    ans = ans.squeeze()
    ans = (ans >= 0).all(axis=-1) * ans.sum(axis=-1)
    ans = np.max(ans, axis=0)
    ans = np.maximum(1 - ans, 0)
    return ans


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            #masks = [poly2rle(segm, img.size[1], img.size[0]) for segm in masks]
            masks = [generate_pyramid_label(img.size[1], img.size[0], np.array(segm, dtype=np.float32).reshape(-1,2)) for segm in masks]
            for mask in masks:
                cv2.imwrite(f'/content/sample_data/{idx}.jpg', mask)
            masks = SegmentationMask(masks, img.size, mode='mask')
            #masks = masks.convert("mask")
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
