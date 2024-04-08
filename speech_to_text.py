import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load any audio file of your choice
speech, rate = librosa.load("/Users/ashutoshbogati/Downloads/voice.wav", sr=16000)

# Ensure mono audio and matching sampling rate
if len(speech.shape) > 1:
    speech = speech.mean(axis=0)
if rate != processor.feature_extractor.sampling_rate:
    raise ValueError(f"Sampling rate of the audio ({rate}) does not match the model's requirements.")

# Tokenize the audio and obtain input values
inputs = processor(speech, return_tensors="pt", padding="longest")
input_values = inputs.input_values

# Store logits (non-normalized predictions)
logits = model(input_values).logits

# Store predicted IDs
predicted_ids = torch.argmax(logits, dim=-1)

# Decode the audio to generate text
transcriptions = processor.batch_decode(predicted_ids)
print(transcriptions[0])
