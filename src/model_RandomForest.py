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
    data_split = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return data_split

def train_random_forest(x_train, y_train, n_estimators=100, random_state=RANDOM_STATE):
    classfi = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    classfi.fit(x_train, y_train)
    return classfi

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
    
def plot_evaluate_model_RF(model, x_test, y_test):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=["Dự đoán tiêu cực", "Dự đoán tích cực"], yticklabels=["Thực tế tiêu cực", "Thực tế tích cực"])
    plt.title("Đánh giá mô hình")
    plt.xlabel("Giá trị dự đoán")
    plt.ylabel("Giá trị thực tế")
    plt.show()
    
def plot_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(4, 5))
    plt.bar(["Độ chính xác"], [accuracy],color='green')
    plt.ylim(0, 1)
    plt.title("Độ chính xác của mô hình")
    plt.ylabel("Tỷ lệ")
    plt.text(0, accuracy + 0.02, f"{accuracy:.2%}", ha='center', fontsize=12)
    plt.show()

def plot_classification_metrics(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True, target_names=["Tiêu cực", "Tích cực"])
    metrics = ['precision', 'recall', 'f1-score']
    classes = ["Tiêu cực", "Tích cực"]
    values = {}
    for metric in metrics:
        metric_values = []
        for class_name in classes:
            metric_values.append(report[class_name][metric])
        values[metric] = metric_values
    x = range(len(classes))
    bar_width = 0.25
    plt.figure(figsize=(8, 5))

    for idx in range(len(metrics)):
        metric = metrics[idx]
        x_positions = []
        for pos in x:
            x_positions.append(pos + idx * bar_width)
        
        plt.bar(
            x_positions,
            values[metric],
            width=bar_width,
            label=metric
        )


    x_tick_positions = []
    for pos in x:
        x_tick_positions.append(pos + bar_width)
    
    plt.xticks(ticks=x_tick_positions, labels=classes)
    plt.ylim(0, 1)
    plt.title("Precision - Recall - F1-score theo lớp")
    plt.legend()
    plt.ylabel("Giá trị")
    plt.show()

if __name__ == "__main__":
    path = "output/data_cleaned.csv"
    
    df = load_data(path)
    x, y, vector = text_vector(df, max_features=MAX_FEATURES)
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    model = train_random_forest(x_train, y_train, random_state=RANDOM_STATE)
    y_pred = model.predict(x_test)

    plot_evaluate_model_RF(model, x_test, y_test)
    plot_accuracy(y_test, y_pred)
    plot_classification_metrics(y_test, y_pred)

    save_model_and_vector(model,vector, model_path="output/random_forest_model.pkl",vector_path="output/vector.pkl")
