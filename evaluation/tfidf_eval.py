import argparse
import json
import time
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ---------- tf‑idf cosine ----------
def _tfidf_cos(a: str, b: str,
               vec: TfidfVectorizer) -> float:
    m = vec.fit_transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0][0])


# ---------- eval ----------
def main(parquet_path: str,
         cand_json: str,
         output_json: str):

    df = pd.read_parquet(parquet_path, columns=['wikidata_id', 'text'])
    txt = dict(zip(df['wikidata_id'], df['text']))

    with open(cand_json, 'r', encoding='utf-8') as f:
        cand = json.load(f)

    vec = TfidfVectorizer(stop_words='english')
    bins = defaultdict(int)
    res, sims = [], []
    t0 = time.time()

    for pair in tqdm(cand, disable=len(cand) < 100, desc='tf‑idf'):
        t1 = txt.get(pair['wikidata_id'], '')
        for wid in pair.get('similar_items', []):
            sim = max(0.0, min(1.0, _tfidf_cos(t1, txt.get(wid, ''), vec)))
            sims.append(sim)
            bins[f'{int(sim*10)/10:.1f}-{(int(sim*10)+1)/10:.1f}'] += 1
            res.append({'wikidata_id': pair['wikidata_id'],
                        'similar_to': wid,
                        'cosine_similarity_tfidf': sim})

    total = len(sims)
    summary = {
        'total_pairs_evaluated': total,
        'average_cosine_similarity': sum(sims) / total if total else 0.0,
        'similarity_distribution': {k: v / total for k, v in bins.items()},
        'evaluation_time_seconds': time.time() - t0,
        'evaluated_pairs': res
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'done: {total} pairs, '
          f'avg {summary["average_cosine_similarity"]:.4f}, '
          f'{summary["evaluation_time_seconds"]:.1f}s')


# ---------- cli ----------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parquet',    required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--output',     required=True)
    a = p.parse_args()

    main(a.parquet, a.candidates, a.output)
