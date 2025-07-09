from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import MAX_FEATURES, TEST_SIZE, RANDOM_STATE

def train_model(df):
    # Vector hóa văn bản
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(df['review'])
    y = df['label']

    # Chia huấn luyện và test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # huấn luyện model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Dự đoán và in log
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model, vectorizer
