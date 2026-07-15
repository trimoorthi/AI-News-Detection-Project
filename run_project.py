import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve

# Set random seed for reproducibility
np.random.seed(42)

# Ensure output paths exist
output_dir = "C:/Users/Devaprasad/Desktop/project"
os.makedirs(output_dir, exist_ok=True)

print("Step 1: Loading Data...")
# Load datasets
fake_path = "C:/Users/Devaprasad/Desktop/project/Dataset/Fake.csv"
true_path = "C:/Users/Devaprasad/Desktop/project/Dataset/True.csv"

df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)

# Label data (1 for Fake, 0 for True)
df_fake['label'] = 1
df_true['label'] = 0

print(f"Fake news articles: {len(df_fake)}")
print(f"True news articles: {len(df_true)}")

# Combine datasets
df_all = pd.concat([df_fake, df_true], ignore_index=True)

# Fill missing text values
df_all['text'] = df_all['text'].fillna('')
df_all['title'] = df_all['title'].fillna('')

# Downsample to 10,000 total balanced samples for quick local execution
print("Downsampling to 10,000 balanced samples...")
df_fake_sub = df_fake.sample(n=5000, random_state=42)
df_true_sub = df_true.sample(n=5000, random_state=42)
df_sub = pd.concat([df_fake_sub, df_true_sub], ignore_index=True)
df_sub['text'] = df_sub['text'].fillna('')
df_sub['title'] = df_sub['title'].fillna('')

print("Step 2: Preprocessing from Scratch...")
# Define standard English stopwords
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
    "for", "with", "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
    "will", "just", "don", "should", "now"
}

def clean_text_scratch(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove non-a-z characters and replace with space
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_remove_stopwords(text):
    cleaned = clean_text_scratch(text)
    # Tokenize manually by splitting
    words = cleaned.split()
    # Remove stopwords and words of length < 2
    tokens = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return tokens

# Apply cleaning to get full cleaned strings (for sklearn vectorizer)
df_sub['clean_text'] = df_sub['text'].apply(clean_text_scratch)
df_sub['tokens'] = df_sub['text'].apply(tokenize_and_remove_stopwords)

print("Step 3: Running Exploratory Data Analysis...")
# Word counts
df_sub['word_count'] = df_sub['clean_text'].apply(lambda x: len(x.split()))

# Word Count Distribution Plot
plt.figure(figsize=(10, 5))
sns.histplot(data=df_sub, x='word_count', hue='label', kde=True, bins=50, palette='viridis', multiple='stack')
plt.title('Article Word Count Distribution by Class (0 = True, 1 = Fake)')
plt.xlabel('Word Count')
plt.ylabel('Count')
plt.xlim(0, 1500)  # Limit X-axis to standard article lengths
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'word_count_distribution.png'), dpi=300)
plt.close()

# Frequency distribution of top 20 words for True vs Fake
def get_top_n_words(token_series, n=20):
    word_counts = {}
    for tokens in token_series:
        for t in tokens:
            word_counts[t] = word_counts.get(t, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:n]

top_fake_words = get_top_n_words(df_sub[df_sub['label'] == 1]['tokens'], 20)
top_true_words = get_top_n_words(df_sub[df_sub['label'] == 0]['tokens'], 20)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Fake Words Plot
fake_words, fake_counts = zip(*top_fake_words)
sns.barplot(x=list(fake_counts), y=list(fake_words), ax=axes[0], palette='Reds_r')
axes[0].set_title('Top 20 Most Frequent Words in Fake News')
axes[0].set_xlabel('Frequency')

# True Words Plot
true_words, true_counts = zip(*top_true_words)
sns.barplot(x=list(true_counts), y=list(true_words), ax=axes[1], palette='Blues_r')
axes[1].set_title('Top 20 Most Frequent Words in True News')
axes[1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_words_frequency.png'), dpi=300)
plt.close()

print("Step 4: Custom TF-IDF Feature Extraction from Scratch...")

class CustomTfidfVectorizer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.idf_ = []
        self.vocab_list = []
        
    def fit(self, token_series):
        # Calculate Document Frequency (DF) for each word
        df_dict = {}
        total_docs = len(token_series)
        
        # Word frequency to select top max_features
        global_word_counts = {}
        for tokens in token_series:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_dict[t] = df_dict.get(t, 0) + 1
            for t in tokens:
                global_word_counts[t] = global_word_counts.get(t, 0) + 1
                
        # Select top max_features based on global word count
        sorted_words = sorted(global_word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = [w[0] for w in sorted_words[:self.max_features]]
        
        # Build vocabulary
        self.vocabulary_ = {word: idx for idx, word in enumerate(top_words)}
        self.vocab_list = top_words
        
        # Compute IDF: idf(t) = log((1 + N) / (1 + df(t))) + 1
        self.idf_ = []
        for word in top_words:
            df_val = df_dict[word]
            idf_val = np.log((1 + total_docs) / (1 + df_val)) + 1
            self.idf_.append(idf_val)
        self.idf_ = np.array(self.idf_)
        return self
        
    def transform(self, token_series):
        total_docs = len(token_series)
        tf_matrix = np.zeros((total_docs, len(self.vocabulary_)))
        
        for doc_idx, tokens in enumerate(token_series):
            # Compute TF (raw term counts)
            for t in tokens:
                if t in self.vocabulary_:
                    word_idx = self.vocabulary_[t]
                    tf_matrix[doc_idx, word_idx] += 1
            
            # Multiply TF by IDF
            tf_matrix[doc_idx, :] = tf_matrix[doc_idx, :] * self.idf_
            
            # L2 normalization for each document
            row_norm = np.linalg.norm(tf_matrix[doc_idx, :])
            if row_norm > 0:
                tf_matrix[doc_idx, :] = tf_matrix[doc_idx, :] / row_norm
                
        return tf_matrix
        
    def fit_transform(self, token_series):
        self.fit(token_series)
        return self.transform(token_series)

# Fit Custom TF-IDF on our downsampled tokens
custom_vectorizer = CustomTfidfVectorizer(max_features=1000)
X_custom = custom_vectorizer.fit_transform(df_sub['tokens'])
y_sub = df_sub['label'].values

# Split data
X_train_c, X_test_c, y_train, y_test = train_test_split(X_custom, y_sub, test_size=0.2, random_state=42)

# Fit Scikit-Learn TfidfVectorizer for validation benchmark
print("Fitting sklearn TfidfVectorizer as a benchmark...")
sklearn_vec = TfidfVectorizer(max_features=1000, stop_words='english')
X_sklearn = sklearn_vec.fit_transform(df_sub['clean_text']).toarray()
X_train_sk, X_test_sk, _, _ = train_test_split(X_sklearn, y_sub, test_size=0.2, random_state=42)

# Check custom vs sklearn correlation
mean_corr = np.mean([np.corrcoef(X_custom[i], X_sklearn[i])[0, 1] for i in range(10)])
print(f"Cosine similarity/correlation between custom and sklearn features (first 10 samples avg): {mean_corr:.4f}")

print("Step 5: Custom Classifier Implementations from Scratch...")

# 5a. Logistic Regression from Scratch
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for epoch in range(self.epochs):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)
            
            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}")
                
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.where(probs >= 0.5, 1, 0)

# 5b. KNN from Scratch
class CustomKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X_test):
        predictions = []
        print("  Running KNN classification from scratch using cosine similarity...")
        dot_products = np.dot(X_test, self.X_train.T)  # Shape: (n_test, n_train)
        
        for i in range(X_test.shape[0]):
            nearest_indices = np.argsort(dot_products[i])[-self.k:]
            nearest_labels = self.y_train[nearest_indices]
            vote = np.round(np.mean(nearest_labels))
            predictions.append(int(vote))
        return np.array(predictions)

print("Training Custom Logistic Regression from scratch...")
custom_lr = CustomLogisticRegression(learning_rate=0.5, epochs=100)
custom_lr.fit(X_train_c, y_train)
y_pred_custom_lr = custom_lr.predict(X_test_c)
acc_custom_lr = accuracy_score(y_test, y_pred_custom_lr)
print(f"Custom Logistic Regression Accuracy: {acc_custom_lr:.4f}")

print("Running Custom KNN from scratch...")
custom_knn = CustomKNN(k=5)
custom_knn.fit(X_train_c, y_train)
y_pred_custom_knn = custom_knn.predict(X_test_c)
acc_custom_knn = accuracy_score(y_test, y_pred_custom_knn)
print(f"Custom KNN Accuracy: {acc_custom_knn:.4f}")

print("Step 6: Training and Benchmarking Scikit-Learn Classifiers...")

sklearn_models = {
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "LogReg": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

results = {}
confusion_matrices = {}
roc_curves = {}
pr_curves = {}

results["Custom_LogReg"] = {
    "accuracy": acc_custom_lr,
    "precision": precision_score(y_test, y_pred_custom_lr),
    "recall": recall_score(y_test, y_pred_custom_lr),
    "f1": f1_score(y_test, y_pred_custom_lr)
}
confusion_matrices["Custom_LogReg"] = confusion_matrix(y_test, y_pred_custom_lr)
fpr_clr, tpr_clr, _ = roc_curve(y_test, custom_lr.predict_proba(X_test_c))
roc_curves["Custom_LogReg"] = (fpr_clr, tpr_clr, auc(fpr_clr, tpr_clr))

results["Custom_KNN"] = {
    "accuracy": acc_custom_knn,
    "precision": precision_score(y_test, y_pred_custom_knn),
    "recall": recall_score(y_test, y_pred_custom_knn),
    "f1": f1_score(y_test, y_pred_custom_knn)
}
confusion_matrices["Custom_KNN"] = confusion_matrix(y_test, y_pred_custom_knn)

# Train sklearn models
for name, model in sklearn_models.items():
    print(f"Training sklearn {name}...")
    model.fit(X_train_sk, y_train)
    preds = model.predict(X_test_sk)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
    print(f"  {name} Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    confusion_matrices[name] = confusion_matrix(y_test, preds)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_sk)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_curves[name] = (fpr, tpr, auc(fpr, tpr))
        p_curve, r_curve, _ = precision_recall_curve(y_test, probs)
        pr_curves[name] = (p_curve, r_curve)

print("Step 7: Plotting Evaluation Results...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
model_names = list(results.keys())
for idx, name in enumerate(model_names):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrices[name]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticklabels(['True News', 'Fake News'])
    ax.set_yticklabels(['True News', 'Fake News'])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_curves.items():
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
plt.close()

metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
metrics_melted = pd.melt(metrics_df, id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(data=metrics_melted, x='Model', y='Score', hue='Metric', palette='muted')
plt.ylim(0.7, 1.0)
plt.title('Model Performance Comparison (Accuracy, Precision, Recall, F1-score)')
plt.ylabel('Score')
plt.xlabel('Model')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
plt.close()

print("All outputs generated successfully!")
for name, metrics in results.items():
    print(f"\n{name} Performance:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")

with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
    f.write("=== Model Performance Metrics Summary ===\n")
    for name, metrics in results.items():
        f.write(f"\nModel: {name}\n")
        for metric, score in metrics.items():
            f.write(f"  {metric}: {score:.4f}\n")
            
print("Execution Finished!")
