from src.unilts import clean_input
from src.model_RandomForest import predict_new_review
import joblib

if __name__ == "__main__":
    MODEL_PATH = "output/random_forest_model.pkl"
    VECTORIZER_PATH = "output/vector.pkl"

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Nhập bình luận (nhập 'exit' để thoát):")
    while True:
        user_input = input(">>> ")

        if user_input.lower() == "exit":
            break

        cleaned_text = clean_input(user_input)

        label = predict_new_review(model, vectorizer, cleaned_text)

        sentiment = "positive" if label else "nagative"

        print(f"=> Dự đoán cảm xúc: {label}\n")
