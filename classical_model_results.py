import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os


import data_processing

def plot_ngram_coefficients(dataset, feature_names, coefficients, text_type, n=18):
    # Get the indices that would sort the coefficients by their absolute values, and select the top n
    top_n_indices_by_abs_value = np.argsort(np.abs(coefficients))[-n:]

    # Use these indices to select the top n coefficients and the corresponding feature names
    top_n_coefficients = coefficients[top_n_indices_by_abs_value]
    top_n_feature_names = np.array(feature_names)[top_n_indices_by_abs_value]

    # Sort the top n by their signed values (not the absolute values), so the negative ones appear at the bottom
    top_n_indices_by_signed_value = np.argsort(top_n_coefficients)
    top_n_feature_names = top_n_feature_names[top_n_indices_by_signed_value]
    top_n_coefficients = top_n_coefficients[top_n_indices_by_signed_value]

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.barh(range(n), top_n_coefficients, align='center')
    plt.yticks(range(n), top_n_feature_names, fontsize=14)
    plt.xlabel('Coefficient', fontsize=16)
    plt.tick_params(axis='x', labelsize=14, length=0)
    plt.tick_params(axis='y', length=0)

    # Save the figure as a high-resolution PDF
    plt.tight_layout()
    plt.savefig(f'results/ngram_coefs_{dataset}_{text_type}.pdf', dpi=600)
    plt.clf()

def print_model_accuracy(y_test, y_pred, text_type, model):
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Achieved {accuracy*100:.2f}% test accuracy in {text_type} classification using {type(model).__name__}.')

def fit_model(model, text_type, dataset, altStart):
    # Load, preprocess, and split data
    X, y = data_processing.preprocess_data(f'data/{dataset}.json', text_type)
    X_train, _, X_test, y_train, _, y_test = data_processing.split_data(X, y)
    
    # Fit an TF-IDF vectorizer on training data
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform the test set using the same vectorizer
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Fit the model on the training set
    model.fit(X_train_tfidf, y_train)

    # Compute accuracy on the test set if called for
    if dataset == 'quora_insincere_hand_labeled' and text_type == 'response' and not altStart:
        y_pred = model.predict(X_test_tfidf)
        print_model_accuracy(y_test, y_pred, text_type, model)

    elif dataset == 'quora_insincere_large_bootstrap' and text_type == 'prompt' and not altStart:
        X_test, y_test = data_processing.preprocess_data(f'data/quora_insincere_hand_labeled.json', text_type)
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        print_model_accuracy(y_test, y_pred, text_type, model)

    elif dataset == 'quora_insincere_large_bootstrap' and text_type == 'prompt' and altStart:
        data = []
        folder_path = "data/AlteredDatasets"
        # Iterate through the files in the folder
        for filename in os.listdir(os.path.abspath(folder_path)):
            file_path = os.path.join(folder_path, filename)

            X_test, y_test = data_processing.preprocess_data(file_path, text_type)
            X_test_tfidf = vectorizer.transform(X_test)
            y_pred = model.predict(X_test_tfidf)

            total_items = len(y_pred)
            count_a = np.count_nonzero(y_pred == 'complied')
            percentage_a = int((count_a / total_items) * 100)
            filename = filename[:-5]
            data.append([filename, percentage_a])
        
        data.sort(key=lambda x: x[1], reverse=True)
        for pair in data:
            print(f"{pair[0]:<100} {pair[1]}")



    if isinstance(model, LogisticRegression) and dataset != 'quora_insincere_hand_labeled' and not altStart:
        # Plot the n-gram coefficients for logistic regression
        plot_ngram_coefficients(dataset, vectorizer.get_feature_names_out(), model.coef_[0], text_type)

def prepareAlternativeDatasets():
    json_file_path = "data/all_hand_labeled.json"
    
    with open("alternativeStarts.json", "r") as json_file:
        altStarts = json.load(json_file)

    for start in altStarts:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for item in data:
            item["prompt"] = start + item["prompt"]
        fileName = "data/AlteredDatasets/" + start + ".json"

        with open(fileName, "w") as json_file:
            json.dump(data, json_file, indent=4)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit_random_forest_on_quora_10k', action='store_true')
    parser.add_argument('--Alternate_Starting_Text', action='store_true')
    args = parser.parse_args()

    # Define the classical models
    lr_responses_model = LogisticRegression(
        C=10,
        max_iter=10000,
        penalty='l2',
        solver='liblinear',
        random_state=0
    )
    lr_prompts_model = LogisticRegression(
        C=10,
        max_iter=10000,
        penalty='l2',
        solver='liblinear',
        random_state=0
    )
    rf_responses_model = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=300,
        random_state=0
    )
    rf_prompts_model = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=3000,
        random_state=0
    )
    if args.Alternate_Starting_Text:
        prepareAlternativeDatasets()
        print('Finished preparing alternative prompts.')
        print()

    if not args.Alternate_Starting_Text:
        # Get results for Table 4's classical model accuracies
        print('[Table 4 (classical models)]')
        print('Calculating classical model accuracies for dataset: Quora Insincere Questions...')
    fit_model(lr_responses_model,   'response', 'quora_insincere_hand_labeled', args.Alternate_Starting_Text)
    fit_model(rf_responses_model,   'response', 'quora_insincere_hand_labeled', args.Alternate_Starting_Text)

    # For the following 2 fits, we use the quora_insincere_large_bootstrap dataset to train the model (using a 70/15/15
    # train/validation/test split). However, we test the models on the entirety of  quora_insincere_hand_labeled
    # dataset, not the test set from the quora_insincere_large_bootstrap dataset.
    fit_model(lr_prompts_model,     'prompt',   'quora_insincere_large_bootstrap', args.Alternate_Starting_Text)

    if args.fit_random_forest_on_quora_10k:
        fit_model(rf_prompts_model, 'prompt',   'quora_insincere_large_bootstrap', args.Alternate_Starting_Text)
    elif not args.Alternate_Starting_Text:
        print('Skipping random forest prompt classifier fitting.')

    if not args.Alternate_Starting_Text:
        print()
        # Get results for Fig. 3's n-gram coefficients
        print('[Fig. 3]')
        print('Calculating n-gram coefficients for dataset: Hand-Labeled...')
    fit_model(lr_responses_model,   'response', 'all_hand_labeled', args.Alternate_Starting_Text)
    fit_model(lr_prompts_model,     'prompt',   'all_hand_labeled', args.Alternate_Starting_Text)

    if not args.Alternate_Starting_Text:
        print()
        # Get results for Fig. 4's n-gram coefficients
        print('[Fig. 4]')
        print('Calculating n-gram coefficients for dataset: Bootstrapped Quora Insincere Questions...')
    fit_model(lr_responses_model,   'response', 'quora_insincere_large_bootstrap', args.Alternate_Starting_Text)
    #fit_model(lr_prompts_model,     'prompt',   'quora_insincere_large_bootstrap')  # this was already done above
    print()

    if not args.Alternate_Starting_Text:
        print('Finished; n-gram coefficients were written to the "results" folder.')

    
