# HumanPulse: Automated Perspective Generation, Question Construction, and Retrieval Evaluation Pipeline

This repository provides an end-to-end pipeline for generating multi-perspective summaries, constructing high-quality question sets, and evaluating retrieval performance for long-form articles. The system integrates OpenAI LLMs for summarization and question generation, GPT-based embeddings, and Pinecone as a vector store. It is designed for research applications in question answering, retrieval-augmented generation (RAG), and evaluation methodology.

---

## 1. Overview

The pipeline supports four major tasks:

1. **Content ingestion**: Scraping or loading long-form text sources (HF blogs, CDC, Medium, etc.).
2. **Perspective generation**: Producing multiple independent summaries (perspectives) using a suite of prompt templates.
3. **Question generation**: Creating generic, targeted, and segment-based questions.
4. **Retrieval evaluation**: Embedding perspectives, storing them in Pinecone, and computing top-1 retrieval accuracy (question → article match).

The resulting dataset includes full content, multiple generated question sets, and multiple generated perspectives, stored in both `.json` and `.csv` formats for downstream experimentation.

---

## 2. Repository Structure

| File                     | Description                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| **`llmGenerator.py`**    | Core LLM generation functions for summaries, perspectives, and questions.                               |
| **`prompts.py`**         | Contains prompt templates for generating multiple perspectives.                                         |
| **`scrapeAq_single.py`** | Orchestrates scraping, question generation, perspective generation, and dataset construction.           |
| **`utils.py`**           | Embedding utilities (OpenAI + SBERT), Pinecone index management, text splitting, and scraping helpers.  |
| **`qandprompt_1.json`**  | Example dataset containing content, questions, and perspectives.                                          |

---

## 3. Environment Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="your_api_key"
export PINECONE_API_KEY="your_pinecone_key"
```

---

## 4. Dataset Construction: Content, Perspectives, and Questions

### 4.1 Running the pipeline

To generate the complete dataset:

```python
from scrapeAq_single import writeQandP
writeQandP()
```

This produces:

* `hf_qandprompt_1.json`
* `hf_qandprompt_1.csv`

Both files contain:

* `title`
* `link`
* `content`
* `genericQuestions`
* `targetQuestions`
* `segmentQuestions`
* `sumarries` (multi-prompt perspectives)

---

### 4.2 Question Generation

`addQuestions()` constructs three categories:

1. **Generic questions**

   * Uses `generate_Gen_Qs(...)` to produce general technical questions.

2. **Targeted questions**

   * *Statistics / numbers*
   * *Methods / results*
   * *Insights / opinions*
     Constructed using `generate_Relevant_Qs(...)` with different specifications.

3. **Segment-based questions**

   * Article is split into three overlapping segments via `splitContextIn3(...)`.
   * Two questions are generated per segment.

All question generation logic is contained in `llmGenerator.py` and is executed automatically inside `writeQandP()`.



---

## 5. Perspective Generation

The system creates **multiple independent summaries** (“perspectives”) using the template suite defined in `prompts.py`.

### 5.1 Multi-Prompt Perspective Generation

`generateSummariesList()` iterates through all templates from:

```python
prompts.createPromptList()
```

Templates cover:

* technical achievements and lessons
* academic synthesis
* terminology-focused summaries
* research-topic extraction
* implementation-focused insights
* statistics-oriented summaries
* identification of tangential viewpoints

These perspectives serve as the basis for embedding and retrieval.



---

## 6. Embedding and Vector Index Construction (Pinecone)

To generate embeddings and store them in Pinecone:

```python
from scrapeAq_single import prepareData
prepareData()
```

This process:

1. Loads all perspectives from `all_perspectives_*.json`.
2. Embeds each summary using `text-embedding-3-small` via `get_embedding(...)`.
3. Creates Pinecone indices (`perspective100`, etc.).
4. Upserts embeddings and metadata (`article index`) into the `perspectives` namespace.

All embedding and Pinecone utilities are implemented in `utils.py`.


---

## 7. Retrieval Evaluation: Top-1 Question → Article Matching

### 7.1 Objective

Evaluate whether a generated question retrieves its correct source article when querying a Pinecone index over perspective embeddings.

### 7.2 Procedure

For each question:

1. **Embed the question**

   ```python
   q_embed = get_embedding(question)
   ```

2. **Query Pinecone**

   ```python
   result = pinecone_index.query(
       vector=q_embed,
       top_k=1,
       namespace="perspectives"
   )
   ```

   The returned metadata includes the predicted `article_index`.

3. **Compare with ground truth**
   If `predicted_index == true_index`, the retrieval is counted as correct.

4. **Compute accuracy**
   Accuracy is evaluated separately for:

   * generic questions
   * targeted questions
   * segmented questions
   * per-perspective type
   * overall performance

### 7.3 Example Evaluation Code

```python
import json
from utils import get_embedding
from pinecone import Pinecone
import os

with open("hf_qandprompt_1.json", "r") as f:
    data = json.load(f)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("perspective100")

correct = 0
total = 0

for article in data:
    true_idx = article["index"]
    for q in article["genericQuestions"]:
        q_embed = get_embedding(q)
        res = index.query(vector=q_embed, top_k=1, namespace="perspectives")

        pred_idx = int(res.matches[0].metadata["index"])
        if pred_idx == true_idx:
            correct += 1
        total += 1

print("Generic question top-1 accuracy:", correct / total)
```

---

The pipeline provides a reproducible methodology for evaluating question-context alignment and analyzing which question types and perspective styles produce the strongest retrieval performance.
