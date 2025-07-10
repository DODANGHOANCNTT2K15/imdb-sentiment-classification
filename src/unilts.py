def clean_input(text):
    text = text.lower()

    cleaned_text = ''
    for char in text:
        if char.isalpha() or char.isspace():
            cleaned_text += char
        else:
            cleaned_text += ' '

    cleaned_text = ' '.join(cleaned_text.split())

    return cleaned_text
