"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.classif_datasets import ClassifEvalDataset, ClassifDataset



@registry.register_builder("artpedia_classif")
class ArtpediaClassifBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClassifDataset
    eval_dataset_cls = ClassifEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/artpedia_classif.yaml"}
    
@registry.register_builder("aqua_classif")
class AquaClassifBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClassifDataset
    eval_dataset_cls = ClassifEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aqua/aqua_classif.yaml"}
