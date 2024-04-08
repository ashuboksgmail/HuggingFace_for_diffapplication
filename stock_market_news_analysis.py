from transformers import pipeline

# Create a sentiment analysis pipeline with the finbert model
sentiment_analysis = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Define a list of stock market headlines
headlines = [
    "Stock market reaches all-time high on strong earnings",
    "Investors remain cautious as uncertainty in the market continues",
    "Tech companies report record-breaking quarterly profits",
    "Market volatility shakes investor confidence",
    "Economic indicators point to potential recession",
]

# Classify the sentiment for each headline
for headline in headlines:
    result = sentiment_analysis(headline)
    sentiment_label = result[0]["label"]
    sentiment_score = result[0]["score"]

    print(f"Headline: {headline}")
    print(f"Predicted Sentiment: {sentiment_label} (Score: {sentiment_score:.4f})\n")



