# Research Paper Search Engine

A search engine for academic papers built on the arXiv metadata snapshot. It lets you search through a large collection of papers using three different retrieval methods and view results through a simple web UI.

The project was built as an Information Retrieval coursework piece and covers the full pipeline: data loading, preprocessing, indexing, retrieval, re-ranking, and evaluation.

## Features

- Three retrieval methods in one pipeline
  - **Boolean** retrieval with AND, OR, NOT operators
  - **TF-IDF** ranking using a custom inverted index
  - **Hybrid BM25 + SBERT** (default): BM25 does first-stage retrieval, SBERT does semantic re-ranking
- Clean text preprocessing with NLTK (tokenization, stopword removal, lemmatization)
- Query term highlighting in titles and abstracts
- Gradio web interface for interactive search
- Built-in evaluation with Precision@k, Recall@k, F1, MAP, nDCG@k, and MRR
- Maps internal document indices back to original arXiv IDs

## How it works

The pipeline has four stages.

**1. Data loading.** The notebook reads the arXiv metadata snapshot (a line-delimited JSON file) and loads it into a pandas DataFrame. By default it loads the first 10,000 papers, but you can change the limit.

**2. Preprocessing.** Each paper's title and abstract are cleaned (lowercased, punctuation stripped, stopwords removed, tokens lemmatized). Categories are normalized and the update year is extracted.

**3. Index creation.** Three indices are built in parallel:
   - An inverted index mapping each term to the documents that contain it (with term frequencies)
   - A BM25 index using `rank_bm25`
   - Dense SBERT embeddings using `all-mpnet-base-v2` from `sentence-transformers`

**4. Retrieval.** Depending on the method you pick:
   - Boolean walks the inverted index and applies set operations
   - TF-IDF scores documents using the inverted index and IDF weights
   - Hybrid runs BM25 to get the top 100 candidates, then re-ranks them with cosine similarity between the SBERT query embedding and document embeddings

## Dataset

The project uses the [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) from Kaggle (the `arxiv-metadata-oai-snapshot.json` file). Download it and place it somewhere accessible to the notebook.

The default path in the notebook is:
```
/content/drive/MyDrive/Colab Notebooks/info.r/arxiv-metadata-oai-snapshot.json
```

Change this to match where you put the file.

## Tech stack

- Python 3
- pandas, numpy
- NLTK (stopwords, wordnet, punkt)
- rank-bm25
- sentence-transformers
- scikit-learn
- Gradio
- IPython

## Getting started

### Option 1: Google Colab (recommended)

The notebook is set up to run in Colab and mount Google Drive for the dataset.

1. Open `IR_ev.ipynb` in Google Colab
2. Upload the arXiv dataset to your Google Drive
3. Update `file_path` in the notebook to point to your file
4. Run the cells top to bottom
5. The Gradio cell will launch a public URL you can open in your browser

### Option 2: Local

1. Clone the repo
   ```bash
   git clone https://github.com/Frostie2k23/Research-Paper-Search-Engine.git
   cd Research-Paper-Search-Engine
   ```

2. Install the dependencies
   ```bash
   pip install pandas nltk rank-bm25 sentence-transformers scikit-learn ipython gradio
   ```

3. Download the arXiv dataset and update `file_path` in the notebook to point to your local file (remove the Google Drive mount cell).

4. Open the notebook in Jupyter and run the cells.

> Note: the SBERT model (`all-mpnet-base-v2`) is around 420MB and will be downloaded on first run. Embedding 10,000 documents takes a few minutes on CPU, much faster on GPU.

## Usage

### Via the Gradio UI

Once the Gradio cell runs, you get a search box and a method dropdown. Type a query, pick a method, and hit Search. Results show title, authors, year, categories, relevance score, and a snippet of the abstract, with your query terms highlighted.

Method options:
- `hybrid` (default) - BM25 + SBERT, best for natural language queries
- `boolean` - use AND, OR, NOT between terms, e.g. `neural AND networks NOT recurrent`
- `tfidf` - classic TF-IDF ranking

### Via code

```python
results = search_papers(
    query="deep learning for medical imaging",
    bm25_index=bm25_index,
    sbert_model=sbert_model,
    doc_embeddings=doc_embeddings,
    df=df,
    inverted_index=inverted_index,
    doc_freq=doc_freq,
    method='hybrid',
    index_to_arxiv=index_to_arxiv
)
```

## Evaluation

The notebook includes an evaluation harness. Supply a list of test queries and a ground truth set of relevant document indices per query, and it will compute:

- Precision@k
- Recall@k
- F1 score
- Average Precision (and MAP across queries)
- nDCG@k
- Reciprocal Rank (and MRR across queries)

Example:
```python
test_queries = ["deep learning papers"]
ground_truth = {
    "deep learning papers": {0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 20, 25, 28, 33, 46, 51, 53, 61, 86, 88}
}

evaluate_ir_system(
    test_queries,
    ground_truth,
    bm25_index, sbert_model, doc_embeddings,
    df, inverted_index, doc_freq,
    method='hybrid', k=100
)
```

## Project structure

```
Research-Paper-Search-Engine/
├── IR_ev.ipynb          # Main notebook (pipeline, UI, evaluation)
├── README.md
└── assets/
    └── SE_Logo.png      # Logo shown in the Gradio UI
```

## Notes and limitations

- The Boolean parser is simple. It does left-to-right evaluation and does not support nested parentheses.
- The default dataset limit is 10,000 papers. Increasing this scales the SBERT encoding step linearly, so budget accordingly.
- Ground truth for evaluation has to be supplied manually, since the arXiv dataset doesn't ship with relevance judgments.
- The notebook expects a logo image at a Google Drive path. If you run locally, update or remove the `gr.Image` line in the Gradio cell.

## License

Add a license of your choice (MIT is a common pick for student projects).
