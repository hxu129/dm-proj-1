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
            vals = line.split()
            emb[vals[0]] = np.asarray(vals[1:], dtype='float32')
    return emb

def _text_vec(txt: str, emb: dict[str, np.ndarray]) -> np.ndarray:
    vecs = [emb[w] for w in txt.split() if w in emb]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100, dtype='float32')

def _minhash(vec: np.ndarray, num_perm: int = 128) -> list[int]:
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
def _build_buckets(entries: list[dict], bands: int, rows: int) -> list[dict]:
    buckets = [{} for _ in range(bands)]
    for idx, e in enumerate(entries):
        sig = e['minhash']
        for b in range(bands):
            key = tuple(sig[b * rows:(b + 1) * rows])
            buckets[b].setdefault(key, []).append(idx)
    return buckets

def _query(sig: list[int], buckets: list[dict], bands: int, rows: int) -> set[int]:
    cands: set[int] = set()
    for b in range(bands):
        key = tuple(sig[b * rows:(b + 1) * rows])
        cands.update(buckets[b].get(key, []))
    return cands

def _num_part(wid: str) -> int:
    m = re.search(r'\d+', wid)
    return int(m.group()) if m else 0

# ---------- pipeline ----------
def main(parquet_a: str,
         parquet_b: str,
         output_json: str,
         glove_path: str,
         bands: int = 32,
         num_perm: int = 128):

    if num_perm % bands != 0:
        sys.exit('ERROR: num_perm must be divisible by bands')
    rows = num_perm // bands

    nltk.download('stopwords', quiet=True)
    t0 = time.time()

    df_a = pd.read_parquet(parquet_a)
    df_b = pd.read_parquet(parquet_b)

    txt_a = df_a['text'].apply(_clean)
    txt_b = df_b['text'].apply(_clean)

    glove = _load_glove(glove_path)

    vec_a = txt_a.apply(lambda t: _text_vec(t, glove))
    vec_b = txt_b.apply(lambda t: _text_vec(t, glove))

    ent_a = [{
        'wikidata_id': df_a['wikidata_id'].iloc[i],
        'minhash': _minhash(vec_a[i], num_perm)
    } for i in range(len(df_a))]

    ent_b = [{
        'wikidata_id': df_b['wikidata_id'].iloc[i],
        'minhash': _minhash(vec_b[i], num_perm)
    } for i in range(len(df_b))]

    buckets_b = _build_buckets(ent_b, bands, rows)

    res = []
    for i, ea in enumerate(tqdm(ent_a, disable=len(ent_a) < 1000)):
        cands = _query(ea['minhash'], buckets_b, bands, rows)
        sims = sorted(
            {ent_b[j]['wikidata_id'] for j in cands},
            key=_num_part
        )
        if sims:
            res.append({
                'wikidata_id': ea['wikidata_id'],
                'similar_items': sims
            })
        if (i + 1) % 1000 == 0:
            print(f'processed {i + 1}/{len(ent_a)}')

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f'done: {len(res)} groups, {time.time() - t0:.1f}s')

# ---------- CLI ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_a', required=True, help='first parquet file')
    p.add_argument('--input_b', required=True, help='second parquet file')
    p.add_argument('--output', required=True, help='output JSON path')
    p.add_argument('--glove', required=True, help='GloVe txt file path')
    p.add_argument('--bands', type=int, default=32, help='number of LSH bands')
    p.add_argument('--perm', type=int, default=128, help='number of MinHash permutations')
    args = p.parse_args()

    main(
        args.input_a,
        args.input_b,
        args.output,
        args.glove,
        bands=args.bands,
        num_perm=args.perm
    )
