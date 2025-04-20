#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
import json
import hashlib
import time
import argparse
import sys
from tqdm import tqdm


# ---------- clean / simhash ----------
def _clean(txt: str) -> str:
    txt = re.sub(r'<.*?>', '', str(txt))
    txt = re.sub(r'[^A-Za-z\s]', '', txt).lower()
    return ' '.join(txt.split())


def _hashfunc(x: bytes) -> bytes:
    return hashlib.md5(x).digest()          # 128‑bit MD5


class Simhash:
    def __init__(self, txt: str, bits: int = 64):
        fb = bits // 8
        v = [0] * bits
        for tok in (txt[i:i + 4] for i in range(max(len(txt) - 3, 1))):
            hv = int.from_bytes(_hashfunc(tok.encode())[-fb:], 'big')
            for i in range(bits):
                v[i] += 1 if (hv >> i) & 1 else -1
        self.value = sum(1 << i for i, vi in enumerate(v) if vi > 0)


# ---------- lsh ----------
def _build_buckets(bins: list[str], bands: int, rows: int) -> list[dict]:
    buckets = [{} for _ in range(bands)]
    for idx, b in enumerate(bins):
        for band in range(bands):
            key = b[band * rows:(band + 1) * rows]
            buckets[band].setdefault(key, []).append(idx)
    return buckets


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _numeric(wid: str) -> int:
    m = re.search(r'\d+', wid)
    return int(m.group()) if m else 0


# ---------- pipeline ----------
def main(parquet_a: str,
         parquet_b: str,
         output_json: str,
         bands: int = 4,
         rows: int = 16,
         hd_th: int = 3):

    if bands * rows != 64:
        sys.exit('bands × rows must equal 64')

    t0 = time.time()
    cols = ['text', 'wikidata_id']

    try:
        df_a = pd.read_parquet(parquet_a, columns=cols)
        df_b = pd.read_parquet(parquet_b, columns=cols)
    except Exception as e:
        sys.exit(f'failed to load parquet: {e}')

    if df_a.empty or df_b.empty:
        sys.exit('input parquet is empty')

    bins_a = [format(Simhash(_clean(t)).value, '064b')
              for t in tqdm(df_a['text'], disable=len(df_a) < 1000, desc='simhash A')]
    bins_b = [format(Simhash(_clean(t)).value, '064b')
              for t in tqdm(df_b['text'], disable=len(df_b) < 1000, desc='simhash B')]

    buckets_b = _build_buckets(bins_b, bands, rows)

    sim = {}
    for i, ba in tqdm(list(enumerate(bins_a)), disable=len(bins_a) < 500, desc='query'):
        for band in range(bands):
            key = ba[band * rows:(band + 1) * rows]
            for j in buckets_b[band].get(key, []):
                if _hamming(ba, bins_b[j]) < hd_th:
                    sim.setdefault(df_a['wikidata_id'][i], set()).add(df_b['wikidata_id'][j])

    res = [
        {
            'wikidata_id': k,
            'similar_items': sorted(v, key=_numeric)
        }
        for k, v in sim.items()
    ]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f'done: {len(res)} groups, {time.time() - t0:.1f}s')


# ---------- cli ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_a', required=True)
    p.add_argument('--input_b', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bands', type=int, default=4)
    p.add_argument('--rows',  type=int, default=16)
    p.add_argument('--hd',    type=int, default=3)
    args = p.parse_args()

    main(args.input_a,
         args.input_b,
         args.output,
         bands=args.bands,
         rows=args.rows,
         hd_th=args.hd)
