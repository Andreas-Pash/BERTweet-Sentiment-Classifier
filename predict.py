from preprocess_module import pre_process, to_feature_vector


def check_input_type(x):
    """
    Check whether x is:
      - a string
      - a list or tuple of strings
      - or neither.

    Returns one of: "string", "list_of_strings", "neither"
    """

    # Case 1: single string
    if isinstance(x, str):
        return "string"

    # Case 2: list or tuple of strings
    elif isinstance(x, (list, tuple)) and all(isinstance(i, str) for i in x):
        return "list_of_strings"

    # Case 3: anything else
    else:
        return f'{type(x).__name__}'
    


def predict_labels(feature_vector, classifier):
    """Assuming preprocessed samples, return their predicted labels from the classifier model."""
    return classifier.classify_many(feature_vector)


def predict_label_from_raw(sample, classifier):
    """Assuming raw text, return its predicted label from the classifier model."""

    if check_input_type(sample) ==  "string":
        return classifier.classify(to_feature_vector(pre_process(sample)))
    
    elif check_input_type(sample) ==  "list_of_strings":
        return [classifier.classify(to_feature_vector(pre_process(text))) for text in sample ]
    
    else:
        raise TypeError(
            f"Invalid input type: expected a string or a list of strings, got {type(sample).__name__}."
        )
    

def predict_label_from_raw_sklearn(sample, clf):
    """Predict label(s) from raw text using a fitted scikit-learn classifier.

    Args:
        sample (str | list[str]): A single text or a list of texts.
        clf: Fitted estimator with a `.predict` method.

    Returns:
        Any | list[Any]: Predicted label or list of labels.

    Raises:
        TypeError: If `sample` is not str or list[str].
    """
    if check_input_type(sample) == 'string':
        return clf.predict([sample])[0]

    elif check_input_type(sample) == 'list_of_strings':
        return [clf.predict([txt])[0] for txt in sample]
    
    else:
        raise TypeError(
            f"Invalid input type: expected a string or a list of strings, got {type(sample).__name__}."
        )
    