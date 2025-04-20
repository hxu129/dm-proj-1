[![Chinese Version](https://img.shields.io/badge/README-中文版本-blue)](README_CN.md)

# Text Similarity Calculation and Evaluation Using LSH

This project implements various Locality-Sensitive Hashing (LSH) techniques (including Bit Sampling, MinHash, and SimHash) to efficiently find potentially similar text documents in large datasets, particularly Wikipedia articles identified by `wikidata_id` in the `wiki40b` dataset. The project also includes evaluation scripts to assess the similarity of candidate pairs found using various standard metrics such as Jaccard similarity, cosine similarity (based on TF-IDF or GloVe word embeddings), and Levenshtein edit distance.

## Project Structure

The project consists of three main directories:

1.  `data/`: Contains scripts for downloading the required datasets.
    * You need to download the dataset from [Hugging Face wiki40b repository](https://huggingface.co/datasets/google/wiki40b/tree/refs%2Fconvert%2Fparquet/en) and place the corresponding parquet files in the test and validation folders respectively. The `merge_parquet.py` script will merge the shard files.
2.  `find_pair/`: Contains scripts implementing different LSH algorithms to find candidate similar pairs.
    * Scripts ending with `_inside.py` are used to find similar pairs **within** a **single** input dataset.
    * Scripts ending with `_between.py` are used to find similar pairs **between two** input datasets (one as a query set, the other as an index set).
    * Techniques used: Bit Sampling, MinHash (requires GloVe support), SimHash.
    * Output: JSON files listing potential similar pairs (containing `wikidata_id` and a list of `similar_items`).
3.  `evaluation/`: Contains scripts for evaluating the similarity of candidate pairs found by the `find_pair` scripts.
    * The scripts implement different similarity metrics: cosine similarity (GloVe), Jaccard similarity, Levenshtein edit distance, cosine similarity (TF-IDF).
    * Input: Original Parquet data files and JSON files containing candidate pairs.
    * Output: JSON files containing evaluation summaries (average similarity, distribution) and detailed scores for each similar pair.

## Environment Requirements

* Python 3.11
* Required Python libraries:
    * `pandas`
    * `numpy`
    * `datasets` (for `get_data.py`)
    * `datasketch` (for MinHash scripts)
    * `nltk` (for Jaccard and some MinHash scripts)
    * `scikit-learn` (for TF-IDF and cosine similarity)
    * `python-Levenshtein` (for Levenshtein evaluation)
    * `tqdm` (for displaying progress bars)
* GloVe word embeddings: Required for `lsh_minhash_*.py` and `cosine_glove_eval.py` scripts. Please download the required GloVe parameter file and provide its path when running the relevant scripts. You can download it here: [https://drive.google.com/file/d/1Qp92johqxDWPIh97AQMV8fHEOA7rbPz3/view?usp=sharing](https://drive.google.com/file/d/1Qp92johqxDWPIh97AQMV8fHEOA7rbPz3/view?usp=sharing).

Python libraries can typically be installed using pip:
```bash
pip install pandas numpy datasets nltk scikit-learn python-Levenshtein tqdm
```

## Quick Start

1.  **Download the data:**
    ```bash
    cd data
    python data/get_data.py
    # This will create ./validation/validation.parquet and ./test/test.parquet files in the current directory
    ```

2.  **Find similar pairs (example: using SimHash within the validation set):**
    ```bash
    python find_pair/lsh_simhash_inside.py \
        --input validation/validation.parquet \
        --output similar_pairs_simhash_validation.json \
        --bands 4 \
        --rows 16 \
        --hd 3
    ```

3.  **Evaluate the found similar pairs (example: using Jaccard similarity):**
    ```bash
    python evaluation/jaccard_eval.py \
        --parquet validation/validation.parquet \
        --candidates similar_pairs_simhash_validation.json \
        --output evaluation_jaccard_validation.json
    ```

## Detailed Usage and Script Parameters

### Data Download

* **`data/get_data.py`**
    * **Purpose:** Download the `wiki40b/en` dataset (validation and test sets) and save them locally.
    * **Parameters:** None. The script automatically saves to `validation/validation.parquet` and `test/test.parquet`.
    * **Usage:** `python data/get_data.py`

### Finding Similar Pairs (`find_pair/`)

* **`lsh_bit_sample_between.py`**
    * **Purpose:** Find similar items between two Parquet files using LSH (Bit Sampling) technique.
    * **Parameters:**
        * `--input_a`: (Required) Path to the first Parquet file (query set).
        * `--input_b`: (Required) Path to the second Parquet file (index set).
        * `--output`: (Required) Path to the output JSON file.
        * `--k`: (Optional) Number of bits to sample (`k_bits`). Default value: `30`.
        * `--bands`: (Optional) Number of bands for LSH. Default value: `10`. (`k` must be divisible by `bands`).
        * `--hd`: (Optional) Maximum Hamming distance threshold. Default value: `3`.
    * **Usage:** `python find_pair/lsh_bit_sample_between.py --input_a file_a.parquet --input_b file_b.parquet --output output.json`

* **`lsh_bit_sample_inside.py`**
    * **Purpose:** Find similar items within a single Parquet file using LSH (Bit Sampling) technique.
    * **Parameters:**
        * `--input`: (Required) Path to the input Parquet file.
        * `--output`: (Required) Path to the output JSON file.
        * `--k`: (Optional) Number of bits to sample (`k_bits`). Default value: `30`.
        * `--bands`: (Optional) Number of bands for LSH. Default value: `10`. (`k` must be divisible by `bands`).
        * `--hd`: (Optional) Maximum Hamming distance threshold. Default value: `3`.
    * **Usage:** `python find_pair/lsh_bit_sample_inside.py --input data.parquet --output output.json`

* **`lsh_minhash_between.py`**
    * **Purpose:** Find similar items between two Parquet files using LSH (MinHash, based on GloVe word embeddings). **Requires GloVe file.**
    * **Parameters:**
        * `--input_a`: (Required) Path to the first Parquet file (query set).
        * `--input_b`: (Required) Path to the second Parquet file (index set).
        * `--glove`: (Required) Path to the GloVe word embeddings text file (e.g., `glove.6B.100d.txt`).
        * `--output`: (Required) Path to the output JSON file.
    * **Usage:** `python find_pair/lsh_minhash_between.py --input_a file_a.parquet --input_b file_b.parquet --output output.json`

* **`lsh_minhash_inside.py`**
    * **Purpose:** Find similar items within a single Parquet file using LSH (MinHash, based on GloVe word embeddings). **Requires GloVe file.**
    * **Parameters:**
        * `--input`: (Required) Path to the input Parquet file.
        * `--output`: (Required) Path to the output JSON file.
        * `--glove`: (Required) Path to the GloVe word embeddings text file (e.g., `glove.6B.100d.txt`).
        * `--bands`: (Optional) Number of bands for LSH. Default value: `32`. (`perm` must be divisible by `bands`).
        * `--perm`: (Optional) Number of permutations for MinHash (`num_perm`). Default value: `128`.
    * **Usage:** `python find_pair/lsh_minhash_inside.py --input data.parquet --output output.json --glove path/to/glove.6B.100d.txt`

* **`lsh_simhash_between.py`**
    * **Purpose:** Find similar items between two Parquet files using LSH (SimHash, 64-bit).
    * **Parameters:**
        * `--input_a`: (Required) Path to the first Parquet file (query set).
        * `--input_b`: (Required) Path to the second Parquet file (index set).
        * `--output`: (Required) Path to the output JSON file.
        * `--bands`: (Optional) Number of bands for LSH. Default value: `4`. (`bands * rows` must equal 64).
        * `--rows`: (Optional) Number of rows per band for LSH. Default value: `16`. (`bands * rows` must equal 64).
        * `--hd`: (Optional) Maximum Hamming distance threshold. Default value: `3`.
    * **Usage:** `python find_pair/lsh_simhash_between.py --input_a file_a.parquet --input_b file_b.parquet --output output.json`

* **`lsh_simhash_inside.py`**
    * **Purpose:** Find similar items within a single Parquet file using LSH (SimHash, 64-bit).
    * **Parameters:**
        * `--input`: (Required) Path to the input Parquet file.
        * `--output`: (Required) Path to the output JSON file.
        * `--bands`: (Optional) Number of bands for LSH. Default value: `4`. (`bands * rows` must equal 64).
        * `--rows`: (Optional) Number of rows per band for LSH. Default value: `16`. (`bands * rows` must equal 64).
        * `--hd`: (Optional) Maximum Hamming distance threshold. Default value: `3`.
    * **Usage:** `python find_pair/lsh_simhash_inside.py --input data.parquet --output output.json`

### Evaluating Similar Pairs (`evaluation/`)

* **`cosine_glove_eval.py`**
    * **Purpose:** Evaluate candidate pairs using cosine similarity based on GloVe word embeddings. **Requires GloVe file.**
    * **Parameters:**
        * `--parquet`: (Required) Path to the original Parquet file containing text data.
        * `--candidates`: (Required) Path to the JSON file generated by the `find_pair` script.
        * `--glove`: (Required) Path to the GloVe word embeddings text file (e.g., `glove.6B.100d.txt`).
        * `--output`: (Required) Path to the JSON file for output evaluation results.
    * **Usage:** `python dm/evaluation/cosine_glove_eval.py --parquet data.parquet --candidates candidates.json --glove path/to/glove.txt --output evaluation.json`

* **`jaccard_eval.py`**
    * **Purpose:** Evaluate candidate pairs using Jaccard similarity based on tokenized words (with stopwords removed).
    * **Parameters:**
        * `--parquet`: (Required) Path to the original Parquet file containing text data.
        * `--candidates`: (Required) Path to the JSON file generated by the `find_pair` script.
        * `--output`: (Required) Path to the JSON file for output evaluation results.
        * `--threshold`: (Optional) Jaccard similarity threshold. Default value: `0.8`.
    * **Usage:** `python dm/evaluation/jaccard_eval.py --parquet data.parquet --candidates candidates.json --output evaluation.json`

* **`levenshtein_eval.py`**
    * **Purpose:** Evaluate candidate pairs using Levenshtein edit distance ratio based on raw text.
    * **Parameters:**
        * `--parquet`: (Required) Path to the original Parquet file containing text data.
        * `--candidates`: (Required) Path to the JSON file generated by the `find_pair` script.
        * `--output`: (Required) Path to the JSON file for output evaluation results.
        * `--threshold`: (Optional) Minimum Levenshtein ratio to include in the evaluation. Default value: `0.0` (include all).
    * **Usage:** `python evaluation/levenshtein_eval.py --parquet data.parquet --candidates candidates.json --output evaluation.json`

* **`tfidf_eval.py`**
    * **Purpose:** Evaluate candidate pairs using cosine similarity based on TF-IDF vectors.
    * **Parameters:**
        * `--parquet`: (Required) Path to the original Parquet file containing text data.
        * `--candidates`: (Required) Path to the JSON file generated by the `find_pair` script.
        * `--output`: (Required) Path to the JSON file for output evaluation results.
    * **Usage:** `python evaluation/tfidf_eval.py --parquet data.parquet --candidates candidates.json --output evaluation.json`
