"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)

import random

def select_random_label_and_index(labels):
    ones = [i for i, val in enumerate(labels) if val == 1]
    zeros = [i for i, val in enumerate(labels) if val == 0]
    
    selected_group = random.choice([ones, zeros]) if ones and zeros else ones or zeros
    selected_index = random.choice(selected_group)
    return labels[selected_index], selected_index

def select_label_and_index(labels, index):
    ones = [i for i, val in enumerate(labels) if val == 1]
    zeros = [i for i, val in enumerate(labels) if val == 0]
    
    if index % 2 == 0 and ones:
        selected_index = ones[0]
    elif index % 2 != 0 and zeros:
        selected_index = zeros[0]
    else:
        return None, None 

    return labels[selected_index], selected_index


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class ClassifDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])
        

        label= ann["label"]

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"],
            "label": torch.tensor(label),
        }


class ClassifEvalDataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.captions = []
        self.image = []
        self.label = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.captions.append(caption)
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
            self.label.append(ann["label"])

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        label = ann["label"]
        
        caption = self.text_processor(ann["caption"])


        return {
            "image": image,
            "text_input": caption,
            "instance_id": ann["instance_id"],
            "label": torch.tensor(label),
            "index": torch.tensor(index),
        }