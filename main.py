from helper_function import load_data, split_data, train_and_evaluate_decision_tree
import pandas as pd


def main():
    # Step 1: Load the dataset
    filepath = "two_target_classification.csv"  # Replace with your dataset path
    print("Loading dataset...")
    data = load_data(filepath)

    # Step 2: Split the dataset
    target_columns = ["target1", "target2"]  # Replace with your target columns
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(data, target_columns)

    # Step 3: Train and evaluate Decision Tree models for each target
    print("Training and evaluating Decision Trees...")
    results = train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test, random_state=42, max_depth=5)

    # Step 4: Display results for each target
    for target, result in results.items():
        metrics = result["metrics"]
        print(f"\nResults for target: {target}")
        print(f"Accuracy: {metrics['Accuracy']}")
        print("Classification Report:")
        print(pd.DataFrame(metrics["Classification Report"]).T)
        print("Confusion Matrix:")
        print(metrics["Confusion Matrix"])


if __name__ == "__main__":
    main()
