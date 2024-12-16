# Semi-IIN

## Download Data

The three datasets (CMU-MOSI, CMU-MOSEI, and AMI) are available from this link: 

## Data Directory

To run our preprocessing and training codes directly, please put the necessary files from downloaded data in separate folders as described below.

```
/data/
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
python code/run.py --dataset mosi(default: mosi, option: [mosei, mosi])
```

## Train with MA && Self-training

```bash
python code/run.py --dataset mosi --retrain(store true)
```

