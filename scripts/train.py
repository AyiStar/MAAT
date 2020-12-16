import sys
import json
import logging
import numpy as np
import pandas as pd

sys.path.append('..')
import pyat

dataset = 'assistment'

# read datasets
train_triplets = pd.read_csv(f'../datasets/{dataset}/train_triplets.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'../datasets/{dataset}/concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'../datasets/{dataset}/metadata.json', 'r'))

# construct
train_data = pyat.TrainDataset(train_triplets, concept_map,
                               metadata['num_train_students'], metadata['num_questions'], metadata['num_concepts'])

config = {
    'learning_rate': 0.002,
    'batch_size': 2048,
    'num_epochs': 100,
    'num_dim': 1,
    'device': 'cpu',
}

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname)s] %(message)s',
)

model = pyat.IRTModel(**config)
model.adaptest_init(train_data)
model.adaptest_train(train_data)
model.adaptest_save('../models/irt/checkpoint.pt')
