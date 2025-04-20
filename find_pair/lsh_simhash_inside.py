import pandas as pd
import numpy as np
import re
import json
import hashlib
import time
import argparse
import sys
from itertools import combinations
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
def _build_buckets(bits: list[str], bands: int, rows: int) -> list[dict]:
    buckets = [{} for _ in range(bands)]
    for idx, b in enumerate(bits):
        for band in range(bands):
            key = b[band * rows:(band + 1) * rows]
            buckets[band].setdefault(key, []).append(idx)
    return buckets


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _num(wid: str) -> int:
    return int(re.sub(r'\D', '', wid) or 0)


# ---------- pipeline ----------
def main(parquet_path: str,
         output_json: str,
         bands: int = 4,
         rows: int = 16,
         hd_th: int = 3):

    if bands * rows != 64:
        sys.exit('bands × rows must equal 64')

    t0 = time.time()
    df = pd.read_parquet(parquet_path, columns=['text', 'wikidata_id'])
    if df.empty:
        sys.exit('input parquet is empty')

    # simhash
    bstr = [format(Simhash(_clean(t)).value, '064b')
            for t in tqdm(df['text'], disable=len(df) < 1000, desc='simhash')]

    # lsh buckets
    buckets = _build_buckets(bstr, bands, rows)

    # candidates
    cand = set()
    for d in buckets:
        for idxs in d.values():
            if len(idxs) > 1:
                cand.update(combinations(sorted(idxs), 2))

    # filter
    sim = {}
    for i, j in cand:
        if _hamming(bstr[i], bstr[j]) < hd_th:
            a, b = df['wikidata_id'][i], df['wikidata_id'][j]
            key, val = (a, b) if _num(a) < _num(b) else (b, a)
            sim.setdefault(key, set()).add(val)

    res = [
        {
            'wikidata_id': k,
            'similar_items': sorted(v, key=_num)
        }
        for k, v in sim.items()
    ]

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f'done: {len(res)} groups, {time.time() - t0:.1f}s')


# ---------- cli ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='parquet file')
    p.add_argument('--output', required=True, help='output json')
    p.add_argument('--bands', type=int, default=4)
    p.add_argument('--rows',  type=int, default=16)
    p.add_argument('--hd',    type=int, default=3)
    args = p.parse_args()

    main(args.input,
         args.output,
         bands=args.bands,
         rows=args.rows,
         hd_th=args.hd)
