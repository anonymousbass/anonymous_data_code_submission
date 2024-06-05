## Instructions for training YOLOv8 and YOLOv9 on GOA v1 Dataset

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Training

To train the model(s) in the paper, run this command:

```train
python3 train.py
```

Change the variable "model" to the desired variant of the YOLOv8 and YOLOv9 models for training

### Evaluation

To evaluate the trained model(s) on GOA v1 test data, run:

```eval
python3 eval.py
```
