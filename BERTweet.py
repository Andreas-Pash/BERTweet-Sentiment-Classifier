import numpy as np

import torch

from sklearn.base import BaseEstimator, TransformerMixin

from transformers import AutoTokenizer, AutoModel

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class BertweetEmbeddings(BaseEstimator, TransformerMixin):
    """scikit-learn transformer producing fixed-size BERTweet [CLS] embeddings.

    Converts an iterable of raw texts into a dense matrix of shape
    ``(n_samples, hidden_size)`` suitable for downstream estimators.
    """
    def __init__(self, 
                 model_name: str = "vinai/bertweet-base",
                 max_length: int = 128,
                 batch_size: int = 16,
                 device: str | None = None,
                 preprocess=None):
        """Configure the embedding transformer.

        Parameters
        ----------
        model_name : str
            Hugging Face model identifier.
        max_length : int
            Maximum sequence length (tokens) per example.
        batch_size : int
            Batch size used during encoding.
        device : str | None
            "cuda" or "cpu"; auto-detected if None.
        preprocess : callable | None
            Optional function ``str -> str`` for light text cleaning.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess = preprocess

        self.tokenizer = None
        self.model = None

    def fit(self, X, y=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

    def _embed_batch(self, texts):
        """Compute [CLS] embeddings for a batch of texts.

        Parameters
        ----------
        texts : list[str]
            Preprocessed input texts.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(batch_size, hidden_size)``.
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            # CLS token embedding (index 0); TODO: Add mean pooling option
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings.cpu().numpy()

    def transform(self, X):
        """Transform texts into a matrix of BERTweet [CLS] embeddings.

        Parameters
        ----------
        X : Iterable[str]
            Raw input texts.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_samples, hidden_size)``
        """
        texts = list(X)
        if self.preprocess is not None:
            texts = [self.preprocess(t) for t in texts]

        if len(texts) == 0:
            hidden_size = self.model.config.hidden_size
            return np.zeros((0, hidden_size), dtype=np.float32)

        all_embeds = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeds.append(self._embed_batch(batch))

        return np.vstack(all_embeds)
    