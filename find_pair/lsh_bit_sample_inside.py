import pandas as pd
import numpy as np
import re
import hashlib
import json
import time
import argparse
import sys
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm


# ---------- text → features → Bit‑Sampling ----------
def _preprocess(txt: str) -> str:
    txt = re.sub(r'[^a-z0-9\s]+', '', str(txt).lower())
    return re.sub(r'\s+', ' ', txt).strip()


def _features(txt: str) -> list[str]:
    return txt.split()


def _bit_sample_hash(feats: list[str], base_bits: int = 128, k_bits: int = 30) -> np.ndarray:
    vec = np.zeros(base_bits, dtype=np.int32)
    for f in feats:
        hv = int.from_bytes(hashlib.md5(f.encode()).digest(), 'big')
        for i in range(base_bits):
            vec[base_bits - 1 - i] += 1 if (hv >> i) & 1 else -1
    bits = (vec > 0).astype(np.uint8)
    idx = np.linspace(0, base_bits - 1, k_bits, dtype=int)
    return bits[idx]


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


# ---------- LSH helpers ----------
def _build_buckets(hashes: list[np.ndarray], bands: int, rows: int) -> list[dict]:
    buckets = [{} for _ in range(bands)]
    for idx, h in enumerate(hashes):
        for b in range(bands):
            key = tuple(h[b * rows:(b + 1) * rows])
            buckets[b].setdefault(key, []).append(idx)
    return buckets


# ---------- pipeline ----------
def main(parquet_path: str,
         output_json: str,
         text_col: str = 'text',
         base_bits: int = 128,
         k_bits: int = 30,
         bands: int = 10,
         hd_th: int = 3):

    assert k_bits % bands == 0, 'k_bits must be divisible by bands'
    rows = k_bits // bands
    t0 = time.time()

    df = pd.read_parquet(parquet_path, columns=[text_col, 'wikidata_id'])
    if df.empty:
        sys.exit('input parquet is empty')

    hashes = [_bit_sample_hash(_features(_preprocess(t)), base_bits, k_bits)
              for t in tqdm(df[text_col], disable=len(df) < 1000)]

    buckets = _build_buckets(hashes, bands, rows)

    cand = set()
    for band_dict in buckets:
        for idxs in band_dict.values():
            if len(idxs) > 1:
                cand.update(combinations(sorted(idxs), 2))

    similar = defaultdict(set)
    for i, j in cand:
        if _hamming(hashes[i], hashes[j]) < hd_th:
            a, b = df['wikidata_id'][i], df['wikidata_id'][j]
            key, val = (a, b) if int(re.sub(r'\D', '', a) or 0) < int(re.sub(r'\D', '', b) or 0) else (b, a)
            similar[key].add(val)

    res = [
        {
            'wikidata_id': k,
            'similar_items': sorted(v, key=lambda x: int(re.sub(r'\D', '', x) or 0))
        }
        for k, v in similar.items()
    ]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f'done: {len(res)} groups, {time.time() - t0:.1f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='parquet file')
    parser.add_argument('--output', required=True, help='output json')
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--bands', type=int, default=10)
    parser.add_argument('--hd', type=int, default=3)
    args = parser.parse_args()

    main(args.input, args.output,
         k_bits=args.k,
         bands=args.bands,
         hd_th=args.hd)
