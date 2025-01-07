



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

"""# 4. Model Configuration"""
model_args = ClassificationArgs(
        sliding_window=True,
        use_early_stopping=True,
        early_stopping_metric="r2",
        early_stopping_metric_minimize=False,
        early_stopping_patience=5,
        num_train_epochs=50,
        learning_rate=2e-5,
        evaluate_during_training=True,
        regression=True,
        train_batch_size=16,
        eval_batch_size=8,
        evaluate_during_training_steps=1000,
        max_seq_length=512,
        no_cache=True,
        no_save=True,
        overwrite_output_dir=True,
        reprocess_input_data=True,
        gradient_accumulation_steps=2,
        save_best_model=True,
    )

# Initialize and train the model
model = ClassificationModel(
    "roberta",
    "roberta-large",
    num_labels=1,
    args=model_args,
    use_cuda=False
    )

model.train_model(train_data, eval_df=validation_data, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(validation_data, r2=r2_score, mse=mean_squared_error, mae=mean_absolute_error)
Results.append([result['r2'], result['mse'], result['mae']])




