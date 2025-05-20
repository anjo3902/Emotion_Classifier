import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def print_performance_metrics(y_true, y_pred, model_name="Model"):
    print(f"\n🔍 Performance Metrics for {model_name}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label=1):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, pos_label=1):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, pos_label=1):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['ham', 'spam']))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def evaluate_models(X, y):
    print("\n🚀 Starting model evaluations...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": "logistic_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "Decision Tree": "decision_tree_model.pkl"
    }

    for model_name, file_name in models.items():
        print(f"\n📁 Checking for model: {file_name}")
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                model = pickle.load(f)
            print(f"✅ Loaded {model_name}. Now predicting...")
            y_pred = model.predict(X_test)
            print_performance_metrics(y_test, y_pred, model_name)
        else:
            print(f"❌ Model file '{file_name}' not found!")

def main():
    print(f"📦 Current directory: {os.getcwd()}")
    print(f"📄 Files in directory: {os.listdir()}")

    print("\n📊 Loading dataset...")
    df = pd.read_csv("cleaned_spam_dataset.csv")
    print(f"✅ Total samples loaded: {len(df)}")

    X_text = df['message']
    y = df['label']  # already binary: 0=ham, 1=spam

    print("\n🔄 Loading TF-IDF vectorizer...")
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("✏️ Transforming text data...")
    X = vectorizer.transform(X_text)

    evaluate_models(X, y)

if __name__ == "__main__":
    main()
