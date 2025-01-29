from transformers import AutoTokenizer, AutoModelForSequenceClassification


def download_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Save the tokenizer and model
    tokenizer.save_pretrained("./models")
    model.save_pretrained("./models")


if __name__ == "__main__":
    download_model()
