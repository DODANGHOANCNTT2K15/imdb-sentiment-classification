def clean_input_user(text, model, vectorizer):
    text = text.lower()

    cleaned_text = ''
    for char in text:
        if char.isalpha() or char.isspace():
            cleaned_text += char
        else:
            cleaned_text += ' '

    cleaned_text = ' '.join(cleaned_text.split())

    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"
