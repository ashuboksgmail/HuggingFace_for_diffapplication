#import pipeline
from transformers import pipeline

# Create a fill-mask pipeline with the DistilBERT model
fill_mask = pipeline("fill-mask", model="distilbert-base-uncased")

# sentence with a masked token
sentence = "I studied software engineering in University. I have been working on software since then. I am a [MASK]."

results = fill_mask(sentence)

# Print the predicted words and their scores
for result in results:
    print(f"Predicted word: {result['token_str']}, Score: {result['score']:.4f}")

