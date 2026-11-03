## Live demo
(https://tomato-leaf-disease-classification-n4rctkhgyqrjfnsaejdstv.streamlit.app/)

## Trained model
(https://huggingface.co/sanskar69/tomato-resnet50/tree/main)

# Tomato Leaf Disease Classification (ResNet50)

Classifies a tomato leaf photo into 10 classes (9 diseases + healthy) using
transfer learning with a pretrained ResNet50, and shows the result in a Streamlit app.

## Results (held-out test set, 500 images)
- Test accuracy: about 95.2%
- Macro F1-score: about 0.95
- Weakest class: Target Spot (often predicted as healthy)

## Method
- Pretrained ResNet50 (ImageNet) + Global Average Pooling + Dropout + Dense softmax
- Stage 1: train the new head with the base frozen
- Stage 2: fine-tune the top ResNet50 layers at a low learning rate
- Input 224x224, ResNet50 preprocess_input, augmentation on training data only
- Split: 10,000 train / 500 validation / 500 test

## Download the trained model
The model file (201 MB) is too large for GitHub:
# Tomato Leaf Disease Classification (ResNet50)

Classifies a tomato leaf photo into 10 classes (9 diseases + healthy) using
transfer learning with a pretrained ResNet50, and shows the result in a Streamlit app.

## Results (held-out test set, 500 images)
- Test accuracy: about 95.2%
- Macro F1-score: about 0.95
- Weakest class: Target Spot (often predicted as healthy)

## Method
- Pretrained ResNet50 (ImageNet) + Global Average Pooling + Dropout + Dense softmax
- Stage 1: train the new head with the base frozen
- Stage 2: fine-tune the top ResNet50 layers at a low learning rate
- Input 224x224, ResNet50 preprocess_input, augmentation on training data only
- Split: 10,000 train / 500 validation / 500 test

## Download the trained model
The model file (201 MB) is too large for GitHub:
https://drive.google.com/file/d/1X67wuCumRU0-eRjeeyZLe0WDIvc5uAYh/view?usp=sharing

Put tomato_resnet50_final.keras inside the models/ folder.

## Run locally (Python 3.11)
    py -3.11 -m venv venv
    venv\Scripts\activate.bat
    pip install -r requirements.txt
    streamlit run app.py

## Limitations
- Trained on single-leaf photos with plain backgrounds; real field photos may be
  misclassified, sometimes with high confidence.
- Only 10 classes; other diseases are not recognised.
- Predictions are model estimates, not a substitute for expert agricultural advice.

Put tomato_resnet50_final.keras inside the models/ folder.

## Run locally (Python 3.11)
    py -3.11 -m venv venv
    venv\Scripts\activate.bat
    pip install -r requirements.txt
    streamlit run app.py

## Limitations
- Trained on single-leaf photos with plain backgrounds; real field photos may be
  misclassified, sometimes with high confidence.
- Only 10 classes; other diseases are not recognised.
- Predictions are model estimates, not a substitute for expert agricultural advice.
