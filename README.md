# Semi-IIN
![pic-art-v2](../paper-pic/pic-art-v2.jpg)
## Environment setup

1. create a new environment using conda or pip (We use Python 3.10.16)
2. `pip install -r requirements.txt`

## Download Data

The three datasets (CMU-MOSI, CMU-MOSEI, and AMI) are available from this link: https://pan.baidu.com/s/1_-D5YO_cblNIbnwDJXo0BA 提取码：dw3x 

## Data Directory

To run our preprocessing and training codes directly, please put the necessary files from downloaded data in separate folders as described below.

```
code/data/
    mosi/together/
        audio
        text
        visual
    mosei/
        audio/hubert-FRA
        text/roberta-4-FRA
        visual
    ami/
        audio
        text
        visual
        
```

After that, you should move the data from ami to mosi/mosei data directory, respectively.

## Train with MA only

```bash
python code/run.py 
  --dataset mosi(default: mosi, option: [mosei, mosi]) 
  --lab_num 1284(for mosei, it is 16326; for mosi, it is 1284)
```

## Train with MA && Self-training

```bash
python code/run.py
  --dataset mosi 
  --retrain(store true) 
  --lab_num 1284(for mosei, it is 16326; for mosi, it is 1284)
```

