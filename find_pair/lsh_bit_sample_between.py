import pandas as pd
import numpy as np
import re
import hashlib
import json
import time
import argparse
import sys
from collections import defaultdict
from tqdm import tqdm


# ---------- text → features → Bit‑Sampling ----------
def _preprocess(txt: str) -> str:
    txt = re.sub(r'[^a-z0-9\s]+', '', str(txt).lower())
    return re.sub(r'\s+', ' ', txt).strip()


def _features(txt: str) -> list[str]:
    return txt.split()


def _bit_sample_hash(feats: list[str],
                     base_bits: int = 128,
                     k_bits: int = 30) -> np.ndarray:
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
def _build_buckets(hashes: list[np.ndarray],
                   bands: int,
                   rows: int) -> list[dict]:
    buckets = [{} for _ in range(bands)]
    for idx, h in enumerate(hashes):
        for b in range(bands):
            key = tuple(h[b * rows:(b + 1) * rows])
            buckets[b].setdefault(key, []).append(idx)
    return buckets


def _numeric(wid: str) -> int:
    return int(re.sub(r'\D', '', wid) or 0)


# ---------- pipeline ----------
def main(parquet_a: str,
         parquet_b: str,
         output_json: str,
         text_col: str = 'text',
         base_bits: int = 128,
         k_bits: int = 30,
         bands: int = 10,
         hd_th: int = 3):

    assert k_bits % bands == 0, 'k_bits must be divisible by bands'
    rows = k_bits // bands
    t0 = time.time()

    df_a = pd.read_parquet(parquet_a, columns=[text_col, 'wikidata_id'])
    df_b = pd.read_parquet(parquet_b, columns=[text_col, 'wikidata_id'])
    if df_a.empty or df_b.empty:
        sys.exit('input parquet is empty')

    hashes_a = [_bit_sample_hash(_features(_preprocess(t)), base_bits, k_bits)
                for t in tqdm(df_a[text_col], disable=len(df_a) < 1000)]
    hashes_b = [_bit_sample_hash(_features(_preprocess(t)), base_bits, k_bits)
                for t in tqdm(df_b[text_col], disable=len(df_b) < 1000)]

    buckets_b = _build_buckets(hashes_b, bands, rows)

    similar = defaultdict(set)
    for i, ha in tqdm(list(enumerate(hashes_a)), disable=len(hashes_a) < 500):
        for band in range(bands):
            key = tuple(ha[band * rows:(band + 1) * rows])
            for j in buckets_b[band].get(key, []):
                if _hamming(ha, hashes_b[j]) < hd_th:
                    similar[df_a['wikidata_id'][i]].add(df_b['wikidata_id'][j])

    res = [
        {
            'wikidata_id': k,
            'similar_items': sorted(v, key=_numeric)
        }
        for k, v in similar.items()
    ]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f'done: {len(res)} groups, {time.time() - t0:.1f}s')


# ---------- CLI ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_a', required=True, help='parquet A (query)')
    parser.add_argument('--input_b', required=True, help='parquet B (index)')
    parser.add_argument('--output', required=True, help='output json')
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--bands', type=int, default=10)
    parser.add_argument('--hd', type=int, default=3)
    args = parser.parse_args()

    main(args.input_a,
         args.input_b,
         args.output,
         k_bits=args.k,
         bands=args.bands,
         hd_th=args.hd)
