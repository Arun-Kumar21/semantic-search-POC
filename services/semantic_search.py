from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

class SemanticEmbedding : 
  def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)

  def get_embeddings(self, sentences):
    encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
      model_output = self.model(**encoded_input)
      embeddings = model_output.last_hidden_state.mean(dim=1)

    return embeddings.numpy()
  