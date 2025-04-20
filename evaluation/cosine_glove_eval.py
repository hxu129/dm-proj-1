import argparse
import json
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ---------- glove ----------
def _load_glove(path: str) -> dict[str, np.ndarray]:
    emb: dict[str, np.ndarray] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            emb[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return emb


# ---------- clean / vector ----------
def _clean(txt: str) -> str:
    txt = re.sub(r'<.*?>', '', str(txt))
    txt = re.sub(r'[^A-Za-z\s]', '', txt).lower()
    return ' '.join(txt.split())


def _vec(txt: str,
         emb: dict[str, np.ndarray],
         dim: int = 100) -> np.ndarray:
    ws = [emb[w] for w in txt.split() if w in emb]
    return np.mean(ws, axis=0) if ws else np.zeros(dim, dtype='float32')


# ---------- cosine eval ----------
def main(parquet_path: str,
         cand_json: str,
         glove_path: str,
         output_json: str):

    t0 = time.time()
    df = pd.read_parquet(parquet_path, columns=['wikidata_id', 'text'])
    txt = dict(zip(df['wikidata_id'], df['text']))

    with open(cand_json, 'r', encoding='utf-8') as f:
        cand = json.load(f)

    glove = _load_glove(glove_path)

    bins = defaultdict(int)
    results = []
    sims = []

    for pair in tqdm(cand, disable=len(cand) < 100):
        v1 = _vec(_clean(txt.get(pair['wikidata_id'], '')), glove)
        for wid in pair.get('similar_items', []):
            v2 = _vec(_clean(txt.get(wid, '')), glove)
            sim = float(cosine_similarity([v1], [v2])[0][0])
            sim = max(0.0, min(1.0, sim))
            sims.append(sim)
            idx = min(int(sim * 10), 9)
            bins[f'{idx/10:.1f}-{(idx+1)/10:.1f}'] += 1
            results.append({
                'wikidata_id': pair['wikidata_id'],
                'similar_to': wid,
                'cosine_similarity_glove': sim
            })

    total = len(sims)
    summary = {
        'total_pairs_evaluated': total,
        'average_cosine_similarity': float(np.mean(sims)) if sims else 0.0,
        'similarity_distribution': {k: v / total for k, v in bins.items()},
        'evaluation_time_seconds': time.time() - t0,
        'evaluated_pairs': results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'done: {total} pairs, '
          f'avg {summary["average_cosine_similarity"]:.4f}, '
          f'{summary["evaluation_time_seconds"]:.1f}s')


# ---------- cli ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--glove', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()

    main(a.parquet, a.candidates, a.glove, a.output)
