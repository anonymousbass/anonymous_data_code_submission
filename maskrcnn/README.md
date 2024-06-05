## Instructions for training Mask R-CNN on GOA v1 Dataset

### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To install detectron2:

```setup2
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Training

To train the model(s) in the paper, run this command:

```train
python3 train.py
```

### Evaluation

To evaluate the trained model on GOA v1 test data, run:

```eval
python3 eval.py
```
