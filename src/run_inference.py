import os
import json
import sys
import traceback
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch


def analyze_sentiment(text, model, tokenizer, max_length=128):
    """
    Analyze sentiment of the input text
    """
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )

        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)

        labels = ["NEGATIVE", "POSITIVE"]
        sentiment = labels[prediction.item()]
        confidence = probabilities[0][prediction.item()].item()

        return sentiment, confidence

    except Exception as e:
        print(f"Error analyzing sentiment: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


def main():
    print("Starting sentiment analysis", file=sys.stderr, flush=True)

    text = os.environ.get("INPUT_TEXT", "Default text for analysis")

    model_directory = os.environ.get("MODEL_DIRECTORY", "/models")

    output = {"input": text, "status": "error", "sentiment": None, "confidence": None}

    try:
        model = DistilBertForSequenceClassification.from_pretrained(model_directory)
        tokenizer = DistilBertTokenizer.from_pretrained(model_directory)

        sentiment, confidence = analyze_sentiment(text, model, tokenizer)
        output.update(
            {
                "input": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "status": "success",
            }
        )

    except Exception as e:
        print("Error during processing:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        output["error"] = str(e)

    os.makedirs("/outputs", exist_ok=True)
    output_path = "/outputs/result.json"

    try:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(
            f"Successfully wrote output to {output_path}", file=sys.stderr, flush=True
        )
    except Exception as write_error:
        print("Error writing output file:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
