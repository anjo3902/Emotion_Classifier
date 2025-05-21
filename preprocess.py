# ------------------- Import Libraries -------------------
import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

# ------------------- Download NLTK Resources -------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ------------------- Load Dataset -------------------
data = pd.read_csv('emotions.csv')  # Your original full dataset
print(f"\nFull dataset shape: {data.shape}")

# ------------------- Reduce Dataset Size to 30,000 (Balanced) -------------------
# Ensure 'label' column is int
data['label'] = data['label'].astype(int)

# Define target dataset size
target_size = 30000
num_classes = data['label'].nunique()
samples_per_class = target_size // num_classes

# Sample data per class
sampled_data = data.groupby('label', group_keys=False).apply(lambda x: x.sample(n=samples_per_class, random_state=42)).reset_index(drop=True)

print(f"\nReduced dataset shape: {sampled_data.shape}")
print(sampled_data['label'].value_counts())

# ------------------- Preprocess Text -------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

sampled_data['clean_text'] = sampled_data['text'].apply(preprocess_text)
print("\nText preprocessing completed!")

# ------------------- Label Mapping -------------------
labels_dict = {
    0: "Sadness",
    1: "Fear",
    2: "Love",
    3: "Guilt",
    4: "Anxiety",
    5: "Confusion"
}
#------------------- Encode Labels -------------------
le = LabelEncoder()
y_encoded = le.fit_transform(sampled_data['label'])
encoded_label_to_emotion = {i: labels_dict[le.inverse_transform([i])[0]] for i in range(len(le.classes_))}

# ------------------- Encode Text Using BERT -------------------
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("\nEncoding text using BERT... (this might take a few minutes)")
X = bert_model.encode(sampled_data['clean_text'].tolist(), show_progress_bar=True)
print("BERT encoding completed! Shape:", X.shape)

# ------------------- Train-Test Split -------------------
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y_encoded, sampled_data['text'].tolist(), test_size=0.2, random_state=42
)
print(f"\nData split completed! Train size: {len(X_train)}, Test size: {len(X_test)}")

# ------------------- Train SVM Models -------------------
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
best_f1 = 0
best_kernel = None
best_model = None

print("\nTraining and evaluating SVM models with different kernels...\n")
for kernel in kernels:
    print(f"--- Training SVM with kernel = '{kernel}' ---")
    svm = SVC(kernel=kernel, decision_function_shape='ovr', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"F1 Score (Macro) for kernel '{kernel}': {f1:.4f}")

    print(f"Classification Report for kernel '{kernel}':")
    print(classification_report(y_test, y_pred, target_names=[encoded_label_to_emotion[i] for i in range(len(encoded_label_to_emotion))]))
    print("-------------------------------------------------\n")

    if f1 > best_f1:
        best_f1 = f1
        best_kernel = kernel
        best_model = svm

print(f"\n===== Best Kernel Summary =====")
print(f"Best kernel based on Macro F1 Score: '{best_kernel}' with F1 = {best_f1:.4f}")
print("===============================")

# ------------------- Save Best Model and Encoders -------------------
with open('svm_model_bert_best.pkl', 'wb') as file:
    pickle.dump(best_model, file)

with open('bert_encoder.pkl', 'wb') as file:
    pickle.dump(bert_model, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

with open('label_to_emotion.pkl', 'wb') as file:
    pickle.dump(encoded_label_to_emotion, file)

print("\n✅ Best model and encoders saved successfully.")

# ------------------- Show Sample Predictions -------------------
print("\n🔍 Sample Predictions on test set (first 10):")
y_pred_best = best_model.predict(X_test)

for i in range(min(10, len(X_test))):
    print(f"Original Text: {text_test[i]}")
    true_emotion = encoded_label_to_emotion[y_test[i]]
    pred_emotion = encoded_label_to_emotion[y_pred_best[i]]
    print(f"True Emotion: {true_emotion}, Predicted Emotion: {pred_emotion}")
    print("---")