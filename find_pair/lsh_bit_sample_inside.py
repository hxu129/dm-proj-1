import pandas as pd
import numpy as np
import re
import json
import time
import argparse
import sys
from collections import defaultdict
from tqdm import tqdm
import nltk
import hashlib

# ---------- clean / embed / minhash ----------
def _clean(txt: str) -> str:
    txt = re.sub(r'<.*?>', '', str(txt))
    txt = re.sub(r'[^a-zA-Z\s]', '', txt).lower()
    return ' '.join(txt.split())

def _load_glove(path: str) -> dict[str, np.ndarray]:
    emb: dict[str, np.ndarray] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            emb[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return emb

def _vec(txt: str, emb: dict[str, np.ndarray]) -> np.ndarray:
    vecs = [emb[w] for w in txt.split() if w in emb]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100, dtype='float32')

def _mh(vec: np.ndarray, num_perm: int = 128) -> list[int]:
    sig = [2**64 - 1] * num_perm
    for v in vec:
        if v == 0:
            continue
        feature = str(v).encode('utf-8')
        for j in range(num_perm):
            h = hashlib.sha1(feature + j.to_bytes(2, 'little')).hexdigest()
            hv = int(h, 16)
            if hv < sig[j]:
                sig[j] = hv
    return sig

# ---------- LSH ----------
def _build(hashes: list[list[int]], bands: int, rows: int) -> list[dict]:
    buckets = [{} for _ in range(bands)]
    for idx, sig in enumerate(hashes):
        for b in range(bands):
            key = tuple(sig[b * rows:(b + 1) * rows])
            buckets[b].setdefault(key, []).append(idx)
    return buckets

def _query(sig: list[int], buckets: list[dict], bands: int, rows: int) -> set[int]:
    c = set()
    for b in range(bands):
        key = tuple(sig[b * rows:(b + 1) * rows])
        c.update(buckets[b].get(key, []))
    return c

def _num(wid: str) -> int:
    m = re.search(r'\d+', wid)
    return int(m.group()) if m else 0

# ---------- pipeline ----------
def main(parquet_path: str,
         output_json: str,
         glove_path: str,
         bands: int = 32,
         num_perm: int = 128):

    rows = num_perm // bands
    if num_perm % bands:
        sys.exit('num_perm must be divisible by bands')

    nltk.download('stopwords', quiet=True)
    t0 = time.time()

    df = pd.read_parquet(parquet_path, columns=['text', 'wikidata_id'])
    if df.empty:
        sys.exit('input parquet is empty')

    clean_txt = df['text'].apply(_clean)
    glove = _load_glove(glove_path)

    vecs = clean_txt.apply(lambda t: _vec(t, glove))
    sigs = [_mh(v, num_perm) for v in tqdm(vecs, disable=len(df) < 1000)]

    buckets = _build(sigs, bands, rows)

    sim = defaultdict(set)
    for i, sig in tqdm(list(enumerate(sigs)), disable=len(sigs) < 500):
        for j in _query(sig, buckets, bands, rows):
            if j <= i:
                continue
            sim[df['wikidata_id'][i]].add(df['wikidata_id'][j])

    res = [
        {
            'wikidata_id': k,
            'similar_items': sorted(v, key=_num)
        } for k, v in sim.items()
    ]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f'done: {len(res)} groups, {time.time() - t0:.1f}s')

# ---------- CLI ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='parquet file')
    p.add_argument('--output', required=True, help='output json')
    p.add_argument('--glove', required=True, help='GloVe txt file')
    p.add_argument('--bands', type=int, default=32)
    p.add_argument('--perm', type=int, default=128)
    args = p.parse_args()

    main(args.input,
         args.output,
         args.glove,
         bands=args.bands,
         num_perm=args.perm)
