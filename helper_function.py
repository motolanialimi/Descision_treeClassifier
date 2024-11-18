import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load data
def load_data(filepath):
    """
    Load dataset from a CSV file.
    Args:
        filepath (str): Path to the dataset file.
    Returns:
        DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

# Function to split data into train and test sets
def split_data(data, target_columns, test_size=0.2, random_state=42):
    """
    Split data into train and test sets for multiple targets.
    Args:
        data (DataFrame): Dataset.
        target_columns (list): List of target column names.
        test_size (float): Proportion of test data.
        random_state (int): Random seed.
    Returns:
        tuple: X_train, X_test, y_train, y_test (each for all targets).
    """
    X = data.drop(columns=target_columns)
    y = data[target_columns]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to train and evaluate a Decision Tree model
def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test, **kwargs):
    """
    Train and evaluate a Decision Tree model for multiple targets.
    Args:
        X_train (DataFrame): Training features.
        y_train (DataFrame): Training targets for all target variables.
        X_test (DataFrame): Test features.
        y_test (DataFrame): Test targets for all target variables.
        **kwargs: Additional parameters for DecisionTreeClassifier.
    Returns:
        dict: Evaluation metrics and trained models for each target.
    """
    results = {}
    
    for target in y_train.columns:
        print(f"Training Decision Tree for target: {target}")
        model = DecisionTreeClassifier(**kwargs)
        model.fit(X_train, y_train[target])

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        metrics = {
            "Accuracy": accuracy_score(y_test[target], y_pred),
            "Classification Report": classification_report(y_test[target], y_pred, output_dict=True),
            "Confusion Matrix": confusion_matrix(y_test[target], y_pred)
        }

        # Store results
        results[target] = {"model": model, "metrics": metrics}

    return results
