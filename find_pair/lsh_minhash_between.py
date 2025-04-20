import pandas as pd
import numpy as np
import re
import json
import time
import argparse
import sys
from collections import defaultdict
from datasketch import MinHash
from tqdm import tqdm
import nltk


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
    m = MinHash(num_perm=num_perm)
    for v in vec:
        m.update(str(v).encode())
    return m.hashvalues.tolist()


# ---------- LSH ----------
def _build_buckets(entries: list[dict],
                   bands: int,
                   rows: int) -> list[dict]:
    buckets = [{} for _ in range(bands)]
    for idx, e in enumerate(entries):
        sig = e['minhash']
        for b in range(bands):
            key = tuple(sig[b * rows:(b + 1) * rows])
            buckets[b].setdefault(key, []).append(idx)
    return buckets


def _query(sig: list[int],
           buckets: list[dict],
           bands: int,
           rows: int) -> set[int]:
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
         output_json: str):

    nltk.download('stopwords', quiet=True)
    t0 = time.time()

    df_a = pd.read_parquet(parquet_a)
    df_b = pd.read_parquet(parquet_b)

    txt_a = df_a['text'].apply(_clean)
    txt_b = df_b['text'].apply(_clean)

    glove = _load_glove(r'glove.6B\glove.6B.100d.txt')

    vec_a = txt_a.apply(lambda t: _text_vec(t, glove))
    vec_b = txt_b.apply(lambda t: _text_vec(t, glove))

    ent_a = [{
        'wikidata_id': df_a['wikidata_id'][i],
        'minhash': _minhash(vec_a[i])
    } for i in range(len(df_a))]

    ent_b = [{
        'wikidata_id': df_b['wikidata_id'][i],
        'minhash': _minhash(vec_b[i])
    } for i in range(len(df_b))]

    bands = 32
    rows = 128 // bands
    buckets_b = _build_buckets(ent_b, bands, rows)

    res = []
    for i, ea in enumerate(tqdm(ent_a, disable=len(ent_a) < 1000)):
        cands = _query(ea['minhash'], buckets_b, bands, rows)
        sims = sorted({ent_b[j]['wikidata_id'] for j in cands}, key=_num_part)
        if sims:
            res.append({'wikidata_id': ea['wikidata_id'], 'similar_items': sims})
        if (i + 1) % 1000 == 0:
            print(f'processed {i + 1}/{len(ent_a)}')

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f'done: {len(res)} groups, {time.time() - t0:.1f}s')


# ---------- CLI ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_a', required=True)
    p.add_argument('--input_b', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    main(args.input_a, args.input_b, args.output)
