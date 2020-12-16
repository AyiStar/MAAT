import sys
import json
import datetime
import logging

import torch
import numpy as np
import pandas as pd

sys.path.append('..')
import pyat

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

dataset = 'assistment'
# read datasets
test_triplets = pd.read_csv(f'../datasets/{dataset}/test_triplets.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'../datasets/{dataset}/concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'../datasets/{dataset}/metadata.json', 'r'))

test_data = pyat.AdapTestDataset(test_triplets, concept_map,
                                 metadata['num_test_students'], metadata['num_questions'], metadata['num_concepts'])

config = {
    'learning_rate': 0.0025,
    'batch_size': 2048,
    'num_epochs': 8,
    'num_dim': 1,
    'device': 'cpu',
}

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname)s] %(message)s',
)

test_length = 50
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

strategies = (pyat.RandomStrategy(), pyat.MAATStrategy(n_candidates=10), )

for strategy in strategies:
    model = pyat.IRTModel(**config)
    model.adaptest_init(test_data)
    model.adaptest_preload('../models/irt/checkpoint.pt')
    test_data.reset()
    pyat.AdapTestDriver.run(model, strategy, test_data, test_length, f'../results/{now}')
