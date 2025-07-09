import pandas as pd
from src.model_LogisticRegression import train_model
from src.unilts import clean_input_user

if __name__ == "__main__":
    df = pd.read_csv("output/data_cleaned.csv")
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    model, vectorizer = train_model(df)

    while True:
        user_input = input("\nNhập một đoạn review (gõ 'exit' để thoát):\n ==> ")
        if user_input.lower() == 'exit':
            print("Thoát chương trình.")
            break   
        result = clean_input_user(user_input, model, vectorizer)
        print(f"Kết quả dự đoán: {result}")