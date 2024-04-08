# Import pipeline
from transformers import pipeline

# Create a question-answering pipeline
assistant = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Define the context and question as a dictionary
context = "James Douglas Morrison (December 8, 1943 – July 3, 1971) was an American singer-songwriter and poet who was the" \
          " lead vocalist of the rock band the Doors. Due to his energetic persona, poetic lyrics, distinctive voice, " \
          "unpredictable and erratic performances, along with the dramatic circumstances surrounding his life and early" \
          " death, Morrison is regarded by music critics and fans as one of the most influential frontmen in rock history." \
          " Since his death, his fame has endured as one of popular culture's top rebellious and oft-displayed icons, " \
          "representing the generation gap and youth counterculture"
question = "What is James's profession?"

# Provide context as a dictionary
qa_input = {
    "question": question,
    "context": context
}

# Use the assistant pipeline to answer the question
answer = assistant(qa_input)

print(f"Answer: '{answer['answer']}', score: {round(answer['score'], 4)}, start: {answer['start']}, end: {answer['end']}")



