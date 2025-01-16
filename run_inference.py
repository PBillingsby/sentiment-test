import os
import json
from transformers import pipeline

TASK_NAME = "sentiment-analysis"
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    text = os.environ.get('INPUT_TEXT', 'Default text for analysis')
    print(f"Text to analyze: {text}")

    try:
        classifier = pipeline(TASK_NAME, model=MODEL, device=-1)
        print("Pipeline initialized successfully.")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    try:
        # Perform inference
        result = classifier(text)
        print(f"Pipeline result: {result}")

        # Format output
        output = {
            'input_text': text,
            'sentiment': result[0]['label'],
            'confidence': float(result[0]['score']),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error during inference: {e}")
        output = {
            'input_text': text,
            'error': str(e),
            'status': 'error'
        }

    # Save output to the designated output directory
    try:
        os.makedirs('/outputs', exist_ok=True)
        output_path = '/outputs/result.json'
        print(f"Saving output to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print("Output saved successfully.")
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    main()