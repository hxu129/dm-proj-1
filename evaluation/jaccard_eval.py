import argparse
import json
import re
import time
from collections import defaultdict

import pandas as pd
from nltk import download, word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm


# ---------- nltk ----------
def _init_nltk() -> set[str]:
    download('punkt')
    download('stopwords')
    return set(stopwords.words('english'))


# ---------- clean / tokenise ----------
def _tokens(txt: str, stop: set[str]) -> set[str]:
    txt = re.sub(r'<.*?>', '', str(txt))
    txt = re.sub(r'[^A-Za-z\s]', '', txt).lower()
    return {w for w in word_tokenize(txt) if w not in stop}


# ---------- jaccard ----------
def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------- pipeline ----------
def main(parquet_path: str,
         cand_json: str,
         output_json: str,
         thresh: float = 0.8):

    stop = _init_nltk()
    t0 = time.time()

    df = pd.read_parquet(parquet_path, columns=['wikidata_id', 'text'])
    txt = dict(zip(df['wikidata_id'], df['text']))

    with open(cand_json, 'r', encoding='utf-8') as f:
        cand = json.load(f)

    bins = defaultdict(int)
    sims, res = [], []

    for pair in tqdm(cand, disable=len(cand) < 100, desc='evaluate'):
        tok1 = _tokens(txt.get(pair['wikidata_id'], ''), stop)
        for wid in pair.get('similar_items', []):
            sim = _jaccard(tok1, _tokens(txt.get(wid, ''), stop))
            sims.append(sim)
            key = '1.0-1.0' if sim == 1.0 else f'{int(sim*10)/10:.1f}-{(int(sim*10)+1)/10:.1f}'
            bins[key] += 1
            res.append({'id1': pair['wikidata_id'], 'id2': wid, 'jaccard_similarity': sim})

    total = len(sims)
    summary = {
        'total_pairs_evaluated': total,
        'average_jaccard_similarity': sum(sims) / total if total else 0.0,
        'threshold': thresh,
        'similarity_distribution': {k: v / total for k, v in bins.items()},
        'evaluation_time_seconds': time.time() - t0,
        'evaluated_pairs': res
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'done: {total} pairs, '
          f'avg {summary["average_jaccard_similarity"]:.4f}, '
          f'{summary["evaluation_time_seconds"]:.1f}s')


# ---------- cli ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parquet',    required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--output',     required=True)
    p.add_argument('--threshold',  type=float, default=0.8)
    a = p.parse_args()

    main(a.parquet, a.candidates, a.output, a.threshold)
