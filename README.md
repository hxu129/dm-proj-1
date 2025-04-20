# 使用 LSH 进行文本相似度计算与评估

本项目实现了多种局部敏感哈希（Locality-Sensitive Hashing, LSH）技术（包括位采样 Bit Sampling、最小哈希 MinHash、相似哈希 SimHash），用于在大型数据集中高效地查找可能相似的文本文档，特别是基于 `wiki40b` 数据集中由 `wikidata_id` 标识的维基百科文章。项目还包含了评估脚本，用以评估找到的候选对的相似度，使用了多种标准度量方法，如 Jaccard 相似度、余弦相似度（基于 TF-IDF 或 GloVe 词嵌入）以及 Levenshtein 编辑距离。

## 项目架构

项目主要包含三个目录：

1.  `data/`：包含下载所需数据集的脚本。
    * 请将下载好的parquet文件分别放在`data/test`和`data/validation`文件夹下，`merge_parquet.py`脚本会合并分片文件。
2.  `find_pair/`：包含实现不同 LSH 算法以查找候选相似对的脚本。
    * 以 `_inside.py` 结尾的脚本用于查找**单个**输入数据集**内部**的相似对。
    * 以 `_between.py` 结尾的脚本用于查找**两个**输入数据集**之间**的相似对（一个作为查询集，另一个作为索引集）。
    * 使用的技术：位采样 (Bit Sampling)、最小哈希 (MinHash, 需 GloVe 支持)、相似哈希 (SimHash)。
    * 输出：JSON 文件，列出潜在的相似对（包含 `wikidata_id` 和 `similar_items` 列表）。
3.  `evaluation/`：包含评估 `find_pair` 脚本找到的候选对相似度的脚本。
    * 脚本实现了不同的相似度度量：余弦相似度 (GloVe)、Jaccard 相似度、Levenshtein 编辑距离、余弦相似度 (TF-IDF)。
    * 输入：原始的 Parquet 数据文件和包含候选对的 JSON 文件。
    * 输出：JSON 文件，包含评估摘要（平均相似度、分布）以及每个相似对的详细得分。

## 环境要求

* Python 3.11
* 必需的 Python 库：
    * `pandas`
    * `numpy`
    * `datasets` (用于 `get_data.py`)
    * `datasketch` (用于 MinHash 脚本)
    * `nltk` (用于 Jaccard 和部分 MinHash 脚本)
    * `scikit-learn` (用于 TF-IDF 和余弦相似度)
    * `python-Levenshtein` (用于 Levenshtein 评估)
    * `tqdm` (用于显示进度条)
* GloVe 词嵌入：`lsh_minhash_*.py` 和 `cosine_glove_eval.py` 脚本需要。请下载所需的 GloVe 文件（例如 `glove.6B.100d.txt`），并在运行相关脚本时提供其路径。可在此处下载：[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

通常可以使用 pip 安装 Python 库：
```bash
pip install pandas numpy datasets datasketch nltk scikit-learn python-Levenshtein tqdm
```


## 快速开始

1.  **下载数据：**
    ```bash
    cd dm/data
    python dm/data/get_data.py
    # 这将在当前目录下创建 ./validation/validation.parquet 和 ./test/test.parquet 文件
    ```

2.  **查找相似对（示例：在验证集内部使用 SimHash）：**
    ```bash
    python dm/find_pair/lsh_simhash_inside.py \
        --input validation/validation.parquet \
        --output similar_pairs_simhash_validation.json \
        --bands 4 \
        --rows 16 \
        --hd 3
    ```

3.  **评估找到的相似对（示例：使用 Jaccard 相似度）：**
    ```bash
    python dm/evaluation/jaccard_eval.py \
        --parquet validation/validation.parquet \
        --candidates similar_pairs_simhash_validation.json \
        --output evaluation_jaccard_validation.json
    ```

## 详细用法与脚本参数

### 数据下载

* **`dm/data/get_data.py`**
    * **目的：** 下载 `wiki40b/en` 数据集（验证集和测试集）并保存在本地。
    * **参数：** 无。脚本会自动保存到 `validation/validation.parquet` 和 `test/test.parquet`。
    * **用法：** `python dm/data/get_data.py`

### 查找相似对 (`dm/find_pair/`)

* **`lsh_bit_sample_between.py`**
    * **目的：** 使用 LSH (Bit Sampling) 技术查找两个 Parquet 文件之间的相似项。
    * **参数：**
        * `--input_a`: (必需) 第一个 Parquet 文件的路径（查询集）。
        * `--input_b`: (必需) 第二个 Parquet 文件的路径（索引集）。
        * `--output`: (必需) 输出 JSON 文件的路径。
        * `--k`: (可选) 采样的位数 (`k_bits`)。默认值：`30`。
        * `--bands`: (可选) LSH 的 band 数量。默认值：`10`。 (`k` 必须能被 `bands` 整除)。
        * `--hd`: (可选) 最大汉明距离阈值。默认值：`3`。
    * **用法：** `python dm/find_pair/lsh_bit_sample_between.py --input_a file_a.parquet --input_b file_b.parquet --output output.json`

* **`lsh_bit_sample_inside.py`**
    * **目的：** 使用 LSH (Bit Sampling) 技术在单个 Parquet 文件内部查找相似项。
    * **参数：**
        * `--input`: (必需) 输入 Parquet 文件的路径。
        * `--output`: (必需) 输出 JSON 文件的路径。
        * `--k`: (可选) 采样的位数 (`k_bits`)。默认值：`30`。
        * `--bands`: (可选) LSH 的 band 数量。默认值：`10`。 (`k` 必须能被 `bands` 整除)。
        * `--hd`: (可选) 最大汉明距离阈值。默认值：`3`。
    * **用法：** `python dm/find_pair/lsh_bit_sample_inside.py --input data.parquet --output output.json`

* **`lsh_minhash_between.py`**
    * **目的：** 使用 LSH (MinHash, 基于 GloVe 词嵌入) 查找两个 Parquet 文件之间的相似项。**需要 GloVe 文件。**
    * **参数：**
        * `--input_a`: (必需) 第一个 Parquet 文件的路径（查询集）。
        * `--input_b`: (必需) 第二个 Parquet 文件的路径（索引集）。
        * `--glove`: (必需) GloVe 词嵌入文本文件的路径 (例如 `glove.6B.100d.txt`)。
        * `--output`: (必需) 输出 JSON 文件的路径。
    * **用法：** `python dm/find_pair/lsh_minhash_between.py --input_a file_a.parquet --input_b file_b.parquet --output output.json`

* **`lsh_minhash_inside.py`**
    * **目的：** 使用 LSH (MinHash, 基于 GloVe 词嵌入) 在单个 Parquet 文件内部查找相似项。**需要 GloVe 文件。**
    * **参数：**
        * `--input`: (必需) 输入 Parquet 文件的路径。
        * `--output`: (必需) 输出 JSON 文件的路径。
        * `--glove`: (必需) GloVe 词嵌入文本文件的路径 (例如 `glove.6B.100d.txt`)。
        * `--bands`: (可选) LSH 的 band 数量。默认值：`32`。 (`perm` 必须能被 `bands` 整除)。
        * `--perm`: (可选) MinHash 的置换数量 (`num_perm`)。默认值：`128`。
    * **用法：** `python dm/find_pair/lsh_minhash_inside.py --input data.parquet --output output.json --glove path/to/glove.6B.100d.txt`

* **`lsh_simhash_between.py`**
    * **目的：** 使用 LSH (SimHash, 64位) 查找两个 Parquet 文件之间的相似项。
    * **参数：**
        * `--input_a`: (必需) 第一个 Parquet 文件的路径（查询集）。
        * `--input_b`: (必需) 第二个 Parquet 文件的路径（索引集）。
        * `--output`: (必需) 输出 JSON 文件的路径。
        * `--bands`: (可选) LSH 的 band 数量。默认值：`4`。 (`bands * rows` 必须等于 64)。
        * `--rows`: (可选) LSH 中每个 band 的行数。默认值：`16`。 (`bands * rows` 必须等于 64)。
        * `--hd`: (可选) 最大汉明距离阈值。默认值：`3`。
    * **用法：** `python dm/find_pair/lsh_simhash_between.py --input_a file_a.parquet --input_b file_b.parquet --output output.json`

* **`lsh_simhash_inside.py`**
    * **目的：** 使用 LSH (SimHash, 64位) 在单个 Parquet 文件内部查找相似项。
    * **参数：**
        * `--input`: (必需) 输入 Parquet 文件的路径。
        * `--output`: (必需) 输出 JSON 文件的路径。
        * `--bands`: (可选) LSH 的 band 数量。默认值：`4`。 (`bands * rows` 必须等于 64)。
        * `--rows`: (可选) LSH 中每个 band 的行数。默认值：`16`。 (`bands * rows` 必须等于 64)。
        * `--hd`: (可选) 最大汉明距离阈值。默认值：`3`。
    * **用法：** `python dm/find_pair/lsh_simhash_inside.py --input data.parquet --output output.json`

### 评估相似对 (`dm/evaluation/`)

* **`cosine_glove_eval.py`**
    * **目的：** 使用基于 GloVe 词嵌入的余弦相似度评估候选对。**需要 GloVe 文件。**
    * **参数：**
        * `--parquet`: (必需) 包含文本数据的原始 Parquet 文件路径。
        * `--candidates`: (必需) 由 `find_pair` 脚本生成的 JSON 文件路径。
        * `--glove`: (必需) GloVe 词嵌入文本文件的路径 (例如 `glove.6B.100d.txt`)。
        * `--output`: (必需) 输出评估结果的 JSON 文件路径。
    * **用法：** `python dm/evaluation/cosine_glove_eval.py --parquet data.parquet --candidates candidates.json --glove path/to/glove.txt --output evaluation.json`

* **`jaccard_eval.py`**
    * **目的：** 使用基于分词（已移除停用词）的 Jaccard 相似度评估候选对。
    * **参数：**
        * `--parquet`: (必需) 包含文本数据的原始 Parquet 文件路径。
        * `--candidates`: (必需) 由 `find_pair` 脚本生成的 JSON 文件路径。
        * `--output`: (必需) 输出评估结果的 JSON 文件路径。
        * `--threshold`: (可选) Jaccard 相似度阈值。默认值：`0.8`。
    * **用法：** `python dm/evaluation/jaccard_eval.py --parquet data.parquet --candidates candidates.json --output evaluation.json`

* **`levenshtein_eval.py`**
    * **目的：** 使用基于原始文本的 Levenshtein 编辑距离比率评估候选对。
    * **参数：**
        * `--parquet`: (必需) 包含文本数据的原始 Parquet 文件路径。
        * `--candidates`: (必需) 由 `find_pair` 脚本生成的 JSON 文件路径。
        * `--output`: (必需) 输出评估结果的 JSON 文件路径。
        * `--threshold`: (可选) 要包含在评估中的最小 Levenshtein 比率。默认值：`0.0`（包含所有）。
    * **用法：** `python dm/evaluation/levenshtein_eval.py --parquet data.parquet --candidates candidates.json --output evaluation.json`

* **`tfidf_eval.py`**
    * **目的：** 使用基于 TF-IDF 向量的余弦相似度评估候选对。
    * **参数：**
        * `--parquet`: (必需) 包含文本数据的原始 Parquet 文件路径。
        * `--candidates`: (必需) 由 `find_pair` 脚本生成的 JSON 文件路径。
        * `--output`: (必需) 输出评估结果的 JSON 文件路径。
    * **用法：** `python dm/evaluation/tfidf_eval.py --parquet data.parquet --candidates candidates.json --output evaluation.json`
