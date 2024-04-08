#import pipeline
from transformers import pipeline

#initializing modal for sentiment analysis (distilbert-base-uncased-finetuned-sst-2-english by default)
sentiment_analysis_classifier = pipeline('sentiment-analysis')
sequence = ["Today is a good day for swimming.", "However my girlfriend is sick", "I dont want to go alone"]

sentiment_analysis_results = sentiment_analysis_classifier(sequence)

for result in sentiment_analysis_results:
  print(f"label: {result['label']}, score: {round(result['score'], 4)}")

