from services.semantic_search import SemanticEmbedding
from services.faiss_index import FaissIdx

import pandas as pd

if __name__=='__main__':
  model = SemanticEmbedding()

  # Example

  ''''
  # Example test case
  a = model.get_embeddings('I love playing minecraft')
  print(a.shape)  # Get dimension of generated embedding
  '''

  index = FaissIdx(model)

  '''
  # Add testing docs
  index.add_doc('I love playing minecraft game')
  index.add_doc('Cricket is outdoor game')

  res = index.search_doc('I have minecraft')
  print(res)
  '''

  # sentence-transformers/all-MiniLM-L6-v2  => [{'I love playing minecraft game': np.float32(26.53442)}, {'Cricket is outdoor game': np.float32(85.84185)}]

  # Test on Dataset
  df = pd.read_csv('./dataset/Text_Similarity_Dataset.csv')
  # print(df.head)

  # print(df.info())

  for idx, row in df.iterrows():
    index.add_doc(row['text1'])
  
  res = index.search_doc('portable device made watching media easy')
  print(res)

  # media gadgets get moving pocket-sized devices that let people carry around video and images.. [ With similarities 40.716 ]
  # Time of computation 112 seconds
