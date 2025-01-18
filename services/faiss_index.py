import faiss
import numpy as np

class FaissIdx:
  def __init__(self, model, dim=384):
    self.index = faiss.IndexFlatL2(dim);

    # Maintaining the document data
    self.doc_map = dict()
    self.model = model
    self.ctr = 0 # index counter

  def add_doc(self, doc_text):
    self.index.add(self.model.get_embeddings(doc_text))
    self.doc_map[self.ctr] = doc_text
    self.ctr += 1

  def search_doc(self, query, k=3):
    dist, i = self.index.search(self.model.get_embeddings(query), k)
    return [{self.doc_map[idx] : score} for idx, score in zip(i[0], dist[0]) if idx in self.doc_map]