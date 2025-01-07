



"""# 2. Import Dependencies"""

# After installations, we import the required Python modules for data handling,
# model training, and evaluation.

import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import contractions
from IPython.display import display, Javascript
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
import logging
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
if torch.cuda.is_available():
    # to use GPU
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('GPU is:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



"""# 3. Data Preparation"""

# Read in the data
train_data = pd.read_json('./data.json')


# Sliding window size and overlap
window_size = 500
overlap = 100

# Function to generate sliding window samples
def create_sliding_windows(text, window_size, overlap):
    windows = []
    text=text.split()
    for i in range(0, len(text), overlap):
        window = text[i:i+window_size]
        window = " ".join(window)
        windows.append(window)
    return windows

train_data['sliding_windows'] = train_data['text'].apply(lambda x: create_sliding_windows(x, window_size, overlap))
# print(train_data_df.head())


test_data['sliding_windows'] = test_data['text'].apply(lambda x: create_sliding_windows(x, window_size, overlap))
# print(test_data_df.head())






