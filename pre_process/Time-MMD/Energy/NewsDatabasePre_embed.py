import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
import tqdm
import os
import torch

# model and save dir
BERT_model = 'paraphrase-MiniLM-L6-v2'
save_dir = f'../../../dataset/Time-MMD/textual/Energy/NewsDatabase-embedding-{BERT_model}'
os.makedirs(save_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(BERT_model).to(device)

BATCH_SIZE = 64
ENCODE_BATCH = 32

# read Text DataFrame
df = pd.read_parquet('../../../dataset/Time-MMD/textual/Energy/Energy_combined.parquet')

# transform time format
df['Date_formatted'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

# main progress bar
pbar = tqdm.tqdm(total=len(df), desc="Embedding Text")

texts = []
keys = []

for i, row in df.iterrows():
    texts.append(row['Text'])
    keys.append(row['Date_formatted'])

    if len(texts) == BATCH_SIZE:
        # batch encode
        embeddings = model.encode(texts, batch_size=ENCODE_BATCH, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # save each embedding
        for j in range(len(texts)):
            np.save(os.path.join(save_dir, 'News-' + keys[j] + '.npy'), embeddings[j:j+1])
        pbar.update(len(texts))
        texts, keys = [], []

# last batch
if texts:
    embeddings = model.encode(texts, batch_size=ENCODE_BATCH, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    for j in range(len(texts)):
        np.save(os.path.join(save_dir, 'News-' + keys[j] + '.npy'), embeddings[j:j+1])
    pbar.update(len(texts))

pbar.close()
