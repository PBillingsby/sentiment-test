from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def download_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)

    # Save the tokenizer and model
    tokenizer.save_pretrained('./model')
    model.save_pretrained('./model')

if __name__ == "__main__":
    download_model()