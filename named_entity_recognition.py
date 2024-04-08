#import pipeline
from transformers import pipeline

#import tokenizer to break down text into tokens
from transformers import AutoTokenizer, AutoModelForTokenClassification

#initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

ner_classifier = pipeline("ner", model=model, tokenizer=tokenizer)
sequence = "My name is Ash and I live in Melbourne. I work at Microsoft. "

ner_results = ner_classifier(sequence)

# Define a mapping from entity labels to human-readable labels
label_map = {
    'O': 'Outside of a named entity',
    'B-MIS': 'Miscellaneous entity',
    'I-MIS': 'Miscellaneous entity',
    'B-PER': 'Person',
    'I-PER': 'Person',
    'B-ORG': 'Organization',
    'I-ORG': 'Organization',
    'B-LOC': 'Location',
    'I-LOC': 'Location',
}

print(ner_results)

for result in ner_results:
    entity = result['word']
    entity_label = label_map.get(result['entity'], 'Unknown')
    print(f"{entity}: {entity_label}")

