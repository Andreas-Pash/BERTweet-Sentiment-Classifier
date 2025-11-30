# NLP Tweet Sentiment Classifier

A sentiment analysis project for classifying tweets as **positive** or **negative**, with a focus on handling noisy social media text, negation, sarcasm, and domain-specific language. The project compares a strong **TF–IDF + LinearSVC baseline** against a more advanced **BERTweet-based hybrid model**. 

Author: **Andreas Pashiardis**

---


## Project Overview

This project builds a **tweet sentiment classifier** with the goal of:

* Improving **pre-processing** of noisy Twitter text
* Exploring richer **feature extraction** beyond simple bag-of-words
* Comparing a traditional ML pipeline to a **transformer-based** approach
* Quantifying trade-offs between **accuracy**, **robustness**, and **computational cost**

All evaluation is performed using **10-fold cross-validation**, and scores are reported to **3 decimal places**.

---

## Modeling Approaches

### 1. Rich Pre-processing + TF–IDF + LinearSVC (Baseline)

The baseline is a strong classical NLP pipeline that combines:

#### Pre-processing

* Unicode normalisation
* Lowercasing
* URL and digit removal
* Hashtag splitting into words
* Emoji conversion to text
* Common slang normalisation
* Contraction expansion
 

#### Feature Extraction

* **Word-level TF–IDF**
 
* **Character n-grams**
 
* **Opinion Lexicon features**

  * Counts of positive / negative words from **Bing Liu’s Opinion Lexicon** appended as numeric features

#### Classifier

* **Linear Support Vector Classifier (LinearSVC)** 
---

### 2. BERTweet Hybrid Model

A more advanced **hybrid** architecture leveraging **BERTweet**:

**Dense contextual embeddings:**

  * Tweets are lightly pre-processed
  * Uses the `[CLS]` embedding as a sentence representation

    
**Classifier Pipeline**:

  * **'BERTweet' embeddings**
  * **TF–IDF**
  * **LinearSVC**

This approach improves robustness to **misspellings**, **informal patterns**, and **contextual nuance**, at the cost of higher computational requirements (GPU recommended).

---

## Performance

### Model Comparison (10-fold Cross-Validation)

| Approach                       | Precision | Recall | F1 Score |
| ------------------------------ | --------: | -----: | -------: |
| Rich preprocess + TF–IDF + SVC |     0.886 |  0.887 |    0.886 |
| BERTweet + TF–IDF + SVC        |     0.912 |  0.911 |    0.912 |

**Latency trade-off (approximate per-fold timing):**

* Baseline (TF–IDF + SVC, CPU): ~2 minutes / fold, ~10 minutes total CV
* BERTweet hybrid (GPU): ~10 minutes / fold, ~1 hour total CV
 
---

## Project Structure

```text
.
├── data/
│   ├── raw/              
│   └── lexicon/              
├── src/
│   ├── preprocessing.py   
│   ├── data_parser.py   
│   ├── BERTweet.py  
│   ├── training.py   
│   ├── predict.py     
│   └── metrics.py        
├── notebooks/
│   ├── BaseArchitecture.ipynb
│   ├── BERTweetArchitecture.ipynb
├── reports/
│   └── NLP_Tweet_Classifier_report.pdf
├── requirements.txt
└── README.md
```

---

## Installation & usage

1. **Clone the repository**

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
```

2. **Install dependencies**

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

3. **Training the Baseline Model**

```bash
python train_baseline.py \
  --train_data data/processed/train.csv \
  --model_out models/baseline/model.joblib \
  --vectorizer_out models/baseline/vectorizer.joblib
```

4. **Training the BERTweet Hybrid Model** (recommended only if GPU is available)

```bash
python train_bertweet_hybrid.py \
  --train_data data/processed/train.csv \
  --model_out models/bertweet/model.joblib \
  --device cuda
```

5.  **Running Inference on New Tweets**

```bash
python predict.py \
  --model_path models/bertweet/model.joblib \
  --input_text "I really don't like this app" 
```

 

---

## Future Work

Potential extensions include:

* Explicit **sarcasm detection** or multi-task learning
* Handling **neutral** or multi-class sentiment labels
* You may centralise hyperparameters in a YAML file
* Domain adaptation for specific topics (e.g. politics, product reviews)
* Deployment as:

  * REST API (FastAPI / Flask)
  * Streamlit
* Model distillation or quantization for faster inference on CPU

---

## References

* NLP Tweet Classifier Report – *BERTweet*, Andreas Pashiardis, 2025.
* [Bhagat and Mane, 2020] Bhagat, C. and Mane, D.(2020). Text categorization using sentiment analysis.
  In Proceedings of the International Conference on Computational Science and Applications
* Bing Liu. **Opinion Lexicon**.
* Leung, M. F., Wang, J., and Li, D.(2022). Decentralized robust portfolio optimization
  based on cooperative-competitive multiagent systems.

* Additional works on sentiment analysis and text categorization as listed in the project report.
