import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

# Step 1: Load Dataset
def load_data(path):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['content'] = df['title'] + " " + df['text']
    return df[['content', 'label']]

# Step 2: Preprocess and Split
def preprocess(df):
    print("[INFO] Splitting data...")
    X = df['content']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorize using TF-IDF
def vectorize(X_train, X_test):
    print("[INFO] Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Step 4: Train the model
def train_model(X_train_tfidf, y_train):
    print("[INFO] Training model...")
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_tfidf, y_train)
    return model

# Step 5: Evaluate the model
def evaluate(model, X_test_tfidf, y_test):
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"[RESULT] Accuracy: {acc*100:.2f}%")
    print("[RESULT] Confusion Matrix:\n", cm)

# Step 6: Save model and vectorizer
def save_model(model, vectorizer):
    print("[INFO] Saving model and vectorizer...")
    os.makedirs("model", exist_ok=True)
    with open("model/fake_news_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("[INFO] Saved to 'model/' directory.")

# Main function
def main():
    dataset_path = "dataset/fake_or_real_news.csv"
    df = load_data(dataset_path)
    X_train, X_test, y_train, y_test = preprocess(df)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize(X_train, X_test)
    model = train_model(X_train_tfidf, y_train)
    evaluate(model, X_test_tfidf, y_test)
    save_model(model, vectorizer)

if __name__ == "__main__":
    main()
