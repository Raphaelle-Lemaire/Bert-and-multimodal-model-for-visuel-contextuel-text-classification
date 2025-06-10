# Bert-and-multimodal-model-for-visuel-contextuel-text-classification
This project implements a cross-attention-based architecture to enhance BERT with visual features extracted from a Vision Transformer (ViT). The model classifies sentences (and tokens) from artwork-related texts as either visual (about the painting) or contextual (about external knowledge), without requiring the image at inference.

## LAVIS library
We buid our model with the Lavis library: [link to Lavis](https://github.com/valeriatisch/LAVIS/tree/a154d419ce1fc25de772b6c7309bfb927b557701)
Make sure to clone and install it following their instructions in their documentation: [Lavis documentation](https://opensource.salesforce.com/LAVIS//latest/index.html)

## Repository Organisation
This repository is organized to work with the Lavis library. It also includes example notebooks to test and demonstrate the models.

### Repository for Lavis
The following folders need to be add into the Lavis library following their documentation:

- configs: Yaml files for add new dataset.
- datasets: add for preprocessing datasets (aqua and artpedia).
- projects: Yaml file to run the model.
- tasks: add a new task for classification.

### Example Notebooks
This folder contains Jupyter notebooks to run and test the model.
To use them, place the notebook in the directory of the Lavis library, where you can find eval.py and train.py.

### Link for model weight

