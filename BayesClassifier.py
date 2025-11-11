# Project 1 for IT348
# By Dane Iwema

import random

def print_menu():
    """
    This prints out the menu of initial options for the start of the program
    """
    print("1: Train Classifier")
    print("2: K-fold Validation")
    print("3: Confusion Matrix")
    print("4: Exit\n")

def build(meta_file, k):
    """
    Reads a metadata file and builds k classifiers, each as a dictionary of lists of dictionaries
    for counting feature values per class, initializing all counts to 1.
    Adds a 7th dictionary in each list for counting the number of times
    that class appears (class totals).

    Args:
        meta_file (str): Path to metadata file.
        k (int): Number of classifiers/folds to build. Default is 1.

    Returns:
        counts (list): A list (length k) of dicts, each structured as:
            counts[i][class_label][feature_index][feature_value] = count
    """
    metadata = {}
    with open(meta_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feature_name, values_str = line.split(":")
            values = values_str.split(",")
            metadata[feature_name] = values

    # Extract class labels and remove from features
    class_labels = metadata.pop("class")

    # Build base templates
    feature_template = [{val: 1 for val in values} for values in metadata.values()]
    class_total_template = {cls: 1 for cls in class_labels}

    # Build one base classifier
    def make_classifier():
        classifier = {}
        for cls in class_labels:
            cls_list = [dict(ft) for ft in feature_template]
            cls_list.append(dict(class_total_template))  # add totals
            classifier[cls] = cls_list
        return classifier

    # create k classifiers
    counts = [make_classifier() for _ in range(int(k))]

    return counts

def train(data_file, counts):
    """
    Reads a training data file, updates counts for each classifier (fold),
    and writes the features of each line to temporary fold files (without the target class).
    
    Args:
        data_file (str): Path to training data file
        counts (list of dicts of lists of dicts): classifiers/folds to update
    
    Returns:
        fold_files (list of str): filenames for each fold (e.g., ["fold0.data", ...])
    """
    k = len(counts)
    
    # Open a temporary file for each fold
    fold_files = [f"fold{i}.data" for i in range(k)]
    fold_file_handles = [open(f, "w") for f in fold_files]

    # Process the dataset
    with open(data_file, "r") as data:
        for line in (l for l in data if l.strip()):
            tokens = line.strip().split(",")
            cls = tokens[-1]
            features = tokens[:-1]

            # Pick a random fold/classifier to train on
            model_index = random.randint(0, k - 1)
            model = counts[model_index]

            # Update counts
            model[cls][6][cls] += 1  # class total
            for i, value in enumerate(features):
                model[cls][i][value] += 1

            # Write features only (no target class) to the fold file
            fold_file_handles[model_index].write(",".join(features) + "," + cls + "\n")

    # Close all fold files
    for f in fold_file_handles:
        f.close()

    return fold_files

def collapse_and_precompute(counts):
    """
    Builds k-fold collapsed classifiers and precomputes probabilities P(class) and P(feature|class).

    Args:
        counts (list): list of original classifier count structures
                       (list of dicts of lists of dicts)
    
    Returns:
        pvals (list): list of dicts of lists of dicts mirroring counts but holding probabilities
    """
    k = len(counts)
    if k == 0:
        raise ValueError("Counts list is empty")

    collapsed_classifiers = []
    pvals = []

    # collapse models
    for holdout in range(k):
        # deep copy structure for combined counts
        combined = {cls: [dict(ft) for ft in counts[0][cls]] for cls in counts[0].keys()}

        # sum counts from all folds except the holdout
        for idx, model in enumerate(counts):
            if idx == holdout:
                continue
            for cls in model:
                for i in range(len(model[cls])):
                    for key, val in model[cls][i].items():
                        combined[cls][i][key] += val - 1  # subtract smoothing from double-count

        collapsed_classifiers.append(combined)

    # precompute probabilities
    for model in collapsed_classifiers:
        pmodel = {}
        total_samples = sum(model[next(iter(model))][6].values())  # total of all class counts
        for cls, feature_dicts in model.items():
            pmodel[cls] = []

            # Compute P(class)
            class_total = feature_dicts[6][cls]
            p_class = class_total / total_samples

            # Compute P(feature|class)
            for i, feat_dict in enumerate(feature_dicts[:-1]):
                feature_probs = {}
                for feat_val, count in feat_dict.items():
                    feature_probs[feat_val] = count / class_total
                feature_probs["p"] = p_class if i == 0 else None  # store P(class) in index 0 for reference
                pmodel[cls].append(feature_probs)

            pmodel[cls].append({"p": p_class})

        pvals.append(pmodel)

    return pvals

def predict(features, pmodel):
    """
    Performs the prediction by reading from the features list in the order the features
    appear in the training data and calculates each outcome and then selects the highest
    to return as the predicted outcome

    Args:
        features (list of str) this is the features to use for the predictions
        model (dictionary of lists of dictionaries) the model counts to use for the prediction
    Returns: 
        answer (str) this is the predicted outcome in the form of the target class
    """
    class_probs = {}
    for cls, feature_dicts in pmodel.items():
        # Start with P(class)
        prob = feature_dicts[-1]["p"]

        # Multiply P(feature_i | class)
        for i, feat_val in enumerate(features):
            feat_probs = feature_dicts[i]
            # Use small fallback value if unseen
            p_feat_given_class = feat_probs.get(feat_val, 1e-6)
            prob *= p_feat_given_class

        class_probs[cls] = prob

    # Normalize (optional, but nice to interpret probabilities)
    total = sum(class_probs.values())
    if total > 0:
        for c in class_probs:
            class_probs[c] /= total

    # Pick class with highest probability
    best_class = max(class_probs, key=class_probs.get)
    return best_class

def validate(pmodel, test_file):
    """
    Validates a Naive Bayes model on a test dataset.
    
    Computes accuracy, precision, recall, and F1 score.

    Args:
        pmodel (dict): precomputed model with P(class) and P(feature|class)
        test_file (str): path to test data file (contains features + true class)

    Returns:
        metrics (dict): {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float
        }
        true_labels (list of str): list of all of the actual target class labes
        pred_labels (list of str): list of all predicted target class labels
    """
    # Collect true and predicted labels
    true_labels = []
    pred_labels = []

    with open(test_file, "r") as f:
        for line in (l for l in f if l.strip()):
            tokens = line.strip().split(",")
            features = tokens[:-1]
            true_class = tokens[-1]
            pred_class = predict(features, pmodel)
            true_labels.append(true_class)
            pred_labels.append(pred_class)

    # All classes
    classes = list(pmodel.keys())
    total = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / total if total > 0 else 0.0

    # Compute precision, recall, F1 per class
    precisions = []
    recalls = []
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics = {
        'accuracy': accuracy,
        'precision': sum(precisions) / len(precisions),
        'recall': sum(recalls) / len(recalls),
        'f1': sum(f1s) / len(f1s)
    }
    return metrics, true_labels, pred_labels

# program loop start
classifier={}
running = True
while running:
    print_menu()
    option = input("Option: ")
    match option:
        case "1":
            mDataFile = input("Please input the meta data file path: ")
            tDataFile = input("Please input the training data file path: ")
            classifier = build(mDataFile, 1)
            train(tDataFile, classifier)
            pvalues = collapse_and_precompute(classifier)

        case "2":
            mDataFile = input("Please input the meta data file path: ")
            tDataFile = input("Please input the training data file path: ")
            kfolds = input("Please input a number of k-folds to validate: ")
            kfoldclassifier = build(mDataFile, kfolds)
            fold_files = train(tDataFile, kfoldclassifier)
            pvalues = collapse_and_precompute(kfoldclassifier)
            accuracies = []  # store all validation accuracies
            k = len(pvalues)
            fold_metrics = []
            # perform validation on each fold with the associated unseen data
            for i in range(k):
                test_index = (i - 1) % k
                metrics, _, _ = validate(pvalues[i], fold_files[test_index])
                fold_metrics.append(metrics)
                print(f"Fold {i}: validated on {fold_files[test_index]}")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1 Score:  {metrics['f1']:.4f}\n")
            # add up all metrics and find the average for each
            avg_accuracy = sum(m['accuracy'] for m in fold_metrics) / k
            avg_precision = sum(m['precision'] for m in fold_metrics) / k
            avg_recall = sum(m['recall'] for m in fold_metrics) / k
            avg_f1 = sum(m['f1'] for m in fold_metrics) / k
            print("=== Average k-fold metrics ===")
            print(f"Accuracy:  {avg_accuracy:.4f}")
            print(f"Precision: {avg_precision:.4f}")
            print(f"Recall:    {avg_recall:.4f}")
            print(f"F1 Score:  {avg_f1:.4f}\n\n")

        case "3":
            mDataFile = input("Please input the meta data file path: ")
            trDataFile = input("Please input the training data file path: ")
            tsDataFile = input("Please input the test data file path: ")
            confMatClassifier = build(mDataFile, 1)
            train(trDataFile, confMatClassifier)
            pvalues = collapse_and_precompute(confMatClassifier)
            metrics, true_labels, pred_labels = validate(pvalues[0], tsDataFile)

            classes = list(set(true_labels))  # sorted so the matrix is consistent
            class_to_index = {cls: i for i, cls in enumerate(classes)}

            # initialize the confusion matrix
            n = len(classes)
            conf_matrix = [[0 for _ in range(n)] for _ in range(n)]

            # fFill in the counts
            for true, pred in zip(true_labels, pred_labels):
                i = class_to_index[true]   # row = actual
                j = class_to_index[pred]   # column = predicted
                conf_matrix[i][j] += 1

            # print the confusion matrix
            print("Confusion Matrix:")
            print("Actual \\ Predicted:")
            print("\t", "\t".join(classes))
            for i, row in enumerate(conf_matrix):
                print(classes[i], "\t", "\t".join(str(x) for x in row))

            # print metrics
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}\n")
        case "4":
            running = False
        case _:
            print("Invalid Option")
