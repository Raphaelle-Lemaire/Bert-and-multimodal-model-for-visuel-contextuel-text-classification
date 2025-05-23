"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import logging

import numpy as np
import torch
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("vc_classification")
class Text_V_C_ClassificationTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.inst_id_key = "instance_id"
        self.model=""

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)


        #if hasattr(model, "transformer"):
        #    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.transformer.parameters():
            param.requires_grad = False

        #print("???????????", model.model)
        #for param in model.model.parameters():
        #    param.requires_grad = True
        #for param in model.model.bert.embeddings.word_embeddings.parameters():
        #    param.requires_grad = True
        #for param in model.model.classifier.parameters():
        #    param.requires_grad = True

        return model

    def valid_step(self, model, samples):
        results = []

        outputs = model.predict(samples)

        predictions = outputs["predictions"]
        targets = outputs["targets"]



        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()


        indices = samples[self.inst_id_key]


        for i in range (0,len(samples[self.inst_id_key])):
            index = indices[i]

            if "retrieval" in outputs:
                retrieval = outputs["retrieval"]
                results.append(
                    {
                        self.inst_id_key: int(index),
                        "prediction": predictions[i].tolist(),
                        "target": targets[i].tolist(),
                        "retrieval": retrieval[i].tolist(),
                    }
                )
            else:
                results.append(
                {
                    self.inst_id_key: int(index),
                    "prediction": predictions[i].tolist(),
                    "target": targets[i].tolist(),
                }
            )

        self.model=model
        return results

    def unFreeze(self, agg_metrics, best_agg_metric, split_name, cur_epoch, done):
        done, optimizer = self.model.unFreeze(agg_metrics, best_agg_metric, split_name, cur_epoch, done)
        return done, optimizer

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        self.model.after_evaluation()

        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate=self.inst_id_key,
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        from sklearn.metrics import roc_auc_score, confusion_matrix

        
        results = json.load(open(eval_result_file))

        predictions = np.array([res["prediction"] for res in results])
        targets = np.array([res["target"] for res in results])

        if targets.shape[1]==1:
            targets=targets.squeeze(1)

        predictionPhrase = torch.tensor(predictions, dtype=torch.float32).mean(dim=1)
        targetPhrase = torch.tensor(targets, dtype=torch.float32).mean(dim=1)

        

        predictedBinairePhrase = (predictionPhrase > 0.5).float()

        accuracyPhrase = np.array((predictedBinairePhrase == targetPhrase).float().sum()/ targetPhrase.shape[0])*100
        rocPh = np.array(roc_auc_score(torch.tensor(targetPhrase.squeeze()), torch.tensor(predictedBinairePhrase.squeeze())))*100
        confMatrPhrase = confusion_matrix(targetPhrase.squeeze().flatten(), predictedBinairePhrase.squeeze().flatten())
        
        accuracyMot = (targets == predictions).sum() / targets.shape[0]
        rocMot = np.array(roc_auc_score(torch.tensor(targets.squeeze()), torch.tensor(predictions.squeeze())))*100
        confMatr = confusion_matrix(targets.squeeze().flatten(), predictions.squeeze().flatten())

        mask_0 = targets == 0
        if mask_0.sum() > 0:  
            accuracy_0 = (predictions[mask_0] == 0).sum() / mask_0.sum()
        else:
            accuracy_0 = 0 

        mask_1 = targets == 0
        if mask_1.sum() > 0:  
            accuracy_1 = (predictions[mask_1] == 1).sum() / mask_1.sum()
        else:
            accuracy_1 = 0 
        metrics = {
        "agg_metrics": (accuracyMot+accuracyPhrase.item()+rocMot.item()+rocPh.item()*2)/5,
        "accuracyPhrase": accuracyPhrase.item() ,
        "rocPh": rocPh.item(), 
        "truePositifPhrase+":confMatrPhrase[1,1].item(), 
        "trueNegatifPhrase+":confMatrPhrase[0,0].item(),
        "falsePositifPhrase":confMatrPhrase[0,1].item(),
        "FaulseNegatifPhrase":confMatrPhrase[1,0].item(),
        "accMot": accuracyMot, 
        "rocMot": rocMot.item(), 
        "truePositif+":confMatr[1,1].item(), 
        "trueNegatif+":confMatr[0,0].item(),
        "falsePositif":confMatr[0,1].item(),
        "FaulseNegatif":confMatr[1,0].item()}

        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics

