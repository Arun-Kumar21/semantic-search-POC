# Semantic Search POC

This repository is a **Proof of Concept (POC)** for implementing semantic search using **Hugging Face Transformers** and **FAISS**.

## Overview

Semantic search enhances traditional search capabilities by understanding the meaning of queries and documents rather than relying solely on keyword matching. This POC demonstrates how to:

- Use pre-trained models from Hugging Face to generate embeddings.
- Employ FAISS for efficient similarity search.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Arun-Kumar21/semantic-search-POC.git
cd semantic-search-POC
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Warning**: Running the dataset provided in the dataset file requires a good processor and may take a significant amount of time.

## Key Libraries

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FAISS](https://github.com/facebookresearch/faiss)
