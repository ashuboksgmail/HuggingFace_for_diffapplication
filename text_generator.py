#import pipeline
from transformers import pipeline

text_generator = pipeline('text-generation', model="gpt2")

#starting text
starting_text = "Hi my name is Ash I am a software engineer I am studying in "

# Generate and complete the text for 500 words
completed_text = text_generator(starting_text, max_length=500, do_sample=True, temperature=0.7)

print(completed_text[0]['generated_text'])


