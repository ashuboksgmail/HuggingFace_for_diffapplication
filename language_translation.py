#import pipeline
from transformers import pipeline

#initialize modal to translate english to french
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

text_to_translate = "Together with pianist Ray Manzarek, Morrison founded the Doors in 1965 in " \
                    "Venice, California. The group spent two years in obscurity until shooting to" \
                    " prominence with their number-one hit single in the United States, Light My Fire, " \
                    "taken from their self-titled debut album"
translated_text = translator(text_to_translate, max_length=150)

print(translated_text[0]["translation_text"])
