import os 
import logging

import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2 

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

from preprocess_module import pre_process, light_clean, to_feature_vector
from BERTweet import *
from predict import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def train_classifier(data, architecture = 'base', verbose=False):
    """
    Train a text classifier using either:
      • a TF-IDF + LinearSVC pipeline (architecture='base'), or
      • a BERTweet-embeddings + LinearSVC pipeline (architecture='BERTweet').

    BASE ARCHITECTURE
    --------------------
    Builds a scikit-learn Pipeline consisting of:
        1) FeatureUnion:
              - word-level TF-IDF using tokenizer=pre_process
              - character TF-IDF (3–5 n-grams)
        2) LinearSVC classifier (max_iter=3000)

    BERTWEET ARCHITECTURE
    ----------------------
    Builds a Pipeline consisting of:
        1) FeatureUnion:
              - BertweetEmbeddings(preprocess=light_clean)
              - character TF-IDF (3–5 n-grams)
        2) LinearSVC classifier (balanced class weights)

    Args:
        data (Iterable[(str, Any)]):
            Iterable of (text, label) pairs.
        architecture (str, optional):
            'base'  → TF-IDF + LinearSVC
            'BERTweet' → BERTweet embeddings + LinearSVC
        verbose (bool, optional):
            If True, prints basic training information.

    Returns:
        sklearn.pipeline.Pipeline:
            The trained scikit-learn pipeline.

    Raises:
        ValueError:
            If architecture is not 'base' or 'BERTweet'.
    """
     
    logging.info( f"Training {architecture} Classifier")

    # split data into X (raw text) and y (labels)
    texts  = [x for (x, y) in data]
    labels = [y for (x, y) in data]

    if architecture == 'base':
        vectorizer  = FeatureUnion([
            ('word', TfidfVectorizer(tokenizer=pre_process,
                                    token_pattern=None)),
            ('char', TfidfVectorizer(analyzer='char', ngram_range=(3,5)))
        ])

        pipeline = Pipeline([
            # 1) TF-IDF with controlled vocabulary using pre_process() on words and char (3,5)-grams
            ('vectorizer', vectorizer),
            # 2) Feature selection in TF-IDF space
            # ('select', SelectKBest(chi2, k=10000)),  # keep top 5000 features
            # 3) Classifier
            ('svc', LinearSVC(max_iter=3000))
        ])

        # Fit and return the trained pipeline (pure scikit-learn, no NLTK wrapper)
        return pipeline.fit(texts, labels)


    elif architecture == 'BERTweet':    
        if torch.cuda.is_available():
            if verbose:
                print('Training on a GPU cuda device')        
        else:
            logging.warning('You are attempting to train BERTweet on a CPU device')
            

        vectorizer = FeatureUnion([
        ('bert', BertweetEmbeddings(preprocess=light_clean)),
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5)))
        ])

        BertTweet_SVC = Pipeline([
            ('features', vectorizer),
            ('svc', LinearSVC(
                penalty='l2',
                loss='squared_hinge',
                max_iter=3000,
                class_weight='balanced'
            ))
        ])

        # Fit and return the trained pipeline
        return BertTweet_SVC.fit(texts, labels) 
    else:
        raise ValueError('Either BERTweet or base must be selected')



def evaluate_and_log(classifier, 
                test_data,
                predictions,
                iteration,
                log_dir, 
                metrics_result = None,
                verbose = False
            ):
 
    """
    Write per-fold evaluation details to a text log.

    Groups test samples into TP, TN, FP, FN assuming binary labels
    {"positive","negative"}, and saves counts plus examples alongside
    optional summary metrics.

    Args:
        classifier: Trained model (not used directly; logged for context).
        test_data (Iterable[Tuple[str, Any]]): (text, true_label) pairs.
        predictions (Iterable[Any]): Predicted labels aligned with `test_data`.
        iteration (int): Fold index used in the filename.
        log_dir (str | Path): Directory to create/write the log file.
        metrics_result (Tuple[float, float, float, ...] | None): Optional
            (precision, recall, f1, ...) to include at the top of the log.
        verbose (bool): If True, prints the path to the saved log.

    Logging functionality:
        Creates `log_dir` (if needed) and writes `fold_{iteration}_log.txt`
        containing counts and example tuples for TP/TN/FP/FN.

    Returns:
        None
    """
    # Extract features and labels
    test_texts = [feats for (feats, label) in test_data]
    true_labels = [label for (feats, label) in test_data]
  
    # Initialize counters / containers
    TP, TN, FP, FN = [], [], [], []

    for feats, true_label, pred_label in zip(test_texts, true_labels, predictions):
        if true_label == "positive" and pred_label == "positive":
            TP.append( (feats, true_label, pred_label) ) 
        elif true_label == "negative" and pred_label == "negative":
            TN.append( (feats, true_label, pred_label) ) 
        elif true_label == "negative" and pred_label == "positive":
            FP.append( (feats, true_label, pred_label) ) 
        elif true_label == "positive" and pred_label == "negative":
            FN.append( (feats, true_label, pred_label) ) 
 

    # Create log directory if not exists
    os.makedirs( log_dir , exist_ok=True)
    log_path = f"{log_dir}/fold_{iteration}_log.txt"

    # Write detailed log
    with open(log_path, "w", encoding="utf-8",) as f:
        f.write(f"Fold number {iteration} \n")
        if metrics_result:
            f.write(f"Precision: %f\nRecall: %f\nF-Score: %f" % metrics_result[:3])

        f.write( '(Tweet , True_labels, Predictions) \n\n' )
        f.write(f"True Positives ({len(TP)}):\n{TP}\n\n")
        f.write(f"True Negatives ({len(TN)}):\n{TN}\n\n")
        f.write(f"False Positives ({len(FP)}):\n{FP}\n\n")
        f.write(f"False Negatives ({len(FN)}):\n{FN}\n\n")

    if verbose:
        print(f"Log for iteration {iteration} saved to: {log_path}")
    return 



def cross_validate(dataset, folds, log_dir = None, verbose= False):
    """
    Perform simple K-fold cross-validation over a (text, label) dataset using the
    `train_classifier` pipeline and report averaged metrics.

    The dataset is split into `folds` contiguous slices. For each fold, the slice
    is used as the test set and the remainder as training data. A LinearSVC
    (via `train_classifier`) is fit on the training split, predictions are made
    on the test split with `predict_label_from_raw_sklearn`, and
    `precision_recall_fscore_support(average='weighted')` is recorded.

    Args:
        dataset (Iterable[Tuple[str, Any]]): Sequence of (text, label) pairs.
        folds (int): Number of folds (K).
        log_dir (str | Path | None): If provided, writes per-fold predictions and
            metrics via `evaluate_and_log(...)` for error analysis.
        verbose (bool): If True, prints fold boundaries as training proceeds.

    Returns:
        Tuple[float, float, float, float]:
            (precision, recall, f1, support_avg). With weighted averaging, the
            fourth value (support) is undefined in scikit-learn and will be NaN.

    Notes:
        - Splits are contiguous (not shuffled/stratified).
        - Uses the default `pre_process` inside `train_classifier`.
        - Metrics are averaged across folds with NaNs ignored where applicable.
    """
    results = [] 
    fold_size = int(len(dataset)/folds) + 1
    j = 0

    for i in range(0,len(dataset),int(fold_size)): 
        if verbose:
            print("Fold start on items %d - %d" % ( i, min(len(dataset) -1 , i+fold_size) ) )
        start, end = i, min(len(dataset) -1 , i+fold_size) 
        
        test_idx = np.arange(start, end)
        # training index= everything except val_idx
        train_idx = np.setdiff1d(np.arange(len(dataset)), test_idx) 
        
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]    
        y_test= [label for (text, label) in test_data]
        
        # Training model
        LinearSVC_classifier = train_classifier( train_data )  
        test_preds = [ predict_label_from_raw_sklearn(text, LinearSVC_classifier)  for (text, label) in test_data]    

        result = precision_recall_fscore_support(y_test, test_preds, average='weighted') # evaluate
        results.append( result )  

        # Log it in a file for error analysis 
        if log_dir:
            evaluate_and_log(classifier = LinearSVC_classifier,
                        test_data = test_data,
                        predictions = test_preds,
                        log_dir= log_dir,
                        iteration = j,
                        metrics_result = result
                    )
            j +=1 
    
    # Average the scores
    arr = np.array([[np.nan if v is None else v for v in row] for row in results])
    cross_val_avg_scores = tuple(map(float, np.nanmean(arr, axis=0)))
     
    return  cross_val_avg_scores 

