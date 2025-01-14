



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
from sklearn.manifold import TSNE

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



"""# Model Training With FULL training data (Roberta-large) -- Save Layer 23 -- GPT version (** For final embeddings **)"""

## This code works -- it is reduced, but saves the embeddings from layer 23 onto Drive correctly.
# The model save code here works as well.



# Initialize and train the model
model = ClassificationModel(
    "roberta",
    "roberta-large",
    num_labels=1,
    args=model_args,
    use_cuda=False
    )
    

# Assuming 'text' is the column with text data
all_texts = train_data_df['text'].tolist()


tokenizer = model.tokenizer
device = model.device  # Get the device model is currently on

#only visualizing attention for first text input
inputs = tokenizer(all_texts[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = inputs.to(device)  # Move inputs to the same device as the model
           
outputs = model.model(**inputs, output_hidden_states=True)
attentions = outputs.attentions  # List of attention matrices per layer
attention_matrix = attentions[0][0, 0].detach().numpy()  # Layer 0, Head 0

# Tokenize input for interpretation
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

# Visualize using heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    attention_matrix,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis",
    annot=False,
)
plt.title("Attention Heatmap: Layer 0, Head 0")
plt.xlabel("Keys")
plt.ylabel("Queries")
plt.show()


     




