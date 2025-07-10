import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MAX_FEATURES, TEST_SIZE, RANDOM_STATE
import joblib

def load_data(path):
    df = pd.read_csv(path)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

def text_vector(df, max_features=MAX_FEATURES):
    vector = TfidfVectorizer(max_features=max_features)
    x = vector.fit_transform(df['review'])
    y = df['label']
    return x, y, vector

def split_data(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def train_random_forest(x_train, y_train, n_estimators=100, random_state=RANDOM_STATE):
    classfi = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    classfi.fit(x_train, y_train)
    return classfi

def evaluate_model_RF(model, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nĐộ chính xác: {accuracy:.2%}\n")

    print("Kết quả")
    print(classification_report(y_test, y_pred, target_names=["Tiêu cực", "Tích cực"]))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=["Dự đoán tiêu cực", "Dự đoán tích cực"], yticklabels=["Thực tế tiêu cực", "Thực tế tích cực"])
    plt.title("Đánh giá mô hình")
    plt.xlabel("Giá trị dự đoán")
    plt.ylabel("Giá trị thực tế")
    plt.show()

def save_model_and_vector(model, vector, model_path, vector_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vector, vector_path)
    print(f"Đã lưu: {model_path}")
    print(f"Đã lưu: {vector_path}")

def predict_new_review(model, vector, review_text):
    vector = vector.transform([review_text])
    pred = model.predict(vector)[0]

    if pred == 1:
        return "positive"
    else:
        return "negative"

if __name__ == "__main__":
    path = "output/data_cleaned.csv"
    
    df = load_data(path)
    x, y, vector = text_vector(df, max_features=MAX_FEATURES)
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    model = train_random_forest(x_train, y_train, random_state=RANDOM_STATE)
    evaluate_model_RF(model, x_test, y_test)

    save_model_and_vector(model,vector, model_path="output/random_forest_model.pkl",vector_path="output/vector.pkl")
