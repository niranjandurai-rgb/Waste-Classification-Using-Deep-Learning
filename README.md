# Waste Classification Using Deep Learning (Mini Project)
Six classes: cardboard, glass, metal, paper, plastic, trash.

## Setup
1. `pip install -r requirements.txt`
2. Download TrashNet (`dataset-resized`, one sub-folder per class) from the TrashNet GitHub repo.

## Run
    python train.py --data dataset-resized --model cnn        # baseline
    python train.py --data dataset-resized --model mobilenet  # transfer learning
    streamlit run app.py -- --model outputs/mobilenet/best_model.keras

Outputs per model in `outputs/<model>/`: best_model.keras, metrics.json,
confusion_matrix.png, curves.png. Put the real measured numbers from these in
your report; compare CNN vs MobileNetV2 and discuss confusions (e.g. paper/cardboard).

## Pipeline (matches the survey, Sec. 6)
De-duplicate -> stratified 70/15/15 split -> augmentation (train only, inside model)
-> CNN or MobileNetV2 (frozen, then fine-tuned) -> class-weighted training with
best-val checkpoint + early stopping -> test precision/recall/macro-F1 + confusion
matrix -> Streamlit upload demo with low-confidence warning.
