import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class SoftPromptModel(nn.Module):
    def __init__(self, model_name='gpt2', prompt_length=5):
        super(SoftPromptModel, self).__init__()
        # Load pre-trained GPT-2 model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Define the length of the soft prompt
        self.prompt_length = prompt_length
        
        # Learnable prompt embeddings (size = prompt_length x model_dim)
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, self.model.config.n_embd))
        
    def forward(self, input_text):
        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        
        # Get the input IDs
        input_ids = inputs['input_ids']
        
        # Get input embeddings
        input_embeddings = self.model.transformer.wte(input_ids)  # Token embeddings
        
        # Prepend soft prompt embeddings to the input embeddings
        soft_prompt_embeddings = self.soft_prompt.unsqueeze(0).repeat(input_ids.size(0), 1, 1)
        input_embeddings_with_prompt = torch.cat([soft_prompt_embeddings, input_embeddings], dim=1)
        
        # Forward pass through the GPT-2 model
        outputs = self.model(inputs_embeds=input_embeddings_with_prompt)
        
        return outputs

# Example usage
model = SoftPromptModel()
input_text = "Once upon a time, in a land far away,"
outputs = model(input_text)

# Getting the logits (output of the model)
logits = outputs.logits
print(logits.shape)  # Shape should be (batch_size, sequence_length, vocab_size)