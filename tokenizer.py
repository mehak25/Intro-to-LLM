



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

# Read in training data
train_data = pd.read_json('./train_data.json')





##########Code to implement pre-trained tokenizer########## 

#import AutoTokenizer library  

from transformers import AutoTokenizer  

#initialize tokenizer variable with name of pre-trained model like RoBERTa 

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') 

#loop over every token in a dataframe “text” column and use tokenizer to encode text to LLM tokens 

 tokenized_feature_raw = tokenizer.batch_encode_plus( 

                            # Sentences to encode 

                            train_data.text.values.tolist(),  

                            # Add '[CLS]' and '[SEP]' 

                            add_special_tokens = True       

                   ) 
 print(tokenized_feature_raw)                  
                   
                   
                   
                   
