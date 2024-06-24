# import tensorflow as tf
import shutil
import pandas as pd
import os
from datetime import datetime
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification
# from tensorflow.keras.models import load_model

pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

outputs

###

# Save model into folder w timestamp

current_date = datetime.now().strftime("%d%m%y")

current_dir = os.getcwd()
# Define new folder name
new_folder_name = "distilBert Text Multiclass by 21 Sections caa " +  current_date

# Create the new folder path
new_folder_path = os.path.join(current_dir, new_folder_name)

# Create the new folder if it doesn't already exist
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)
    print(f"Folder '{new_folder_name}' created in {current_dir}")
else:
    print(f"Folder '{new_folder_name}' already exists in {current_dir}")

model.save_pretrained(new_folder_path)
tokenizer.save_pretrained(new_folder_path)

print('large model file created.')

###

def split_file(file_path, chunk_size=50 * 1024 * 1024):  # 100 MB chunks
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        for i in range(0, file_size, chunk_size):
            chunk_data = f.read(chunk_size)
            with open(f"{file_path}.part{i // chunk_size}", 'wb') as chunk_file:
                chunk_file.write(chunk_data)

# Define the paths to your model files
model_file_path = os.path.join(new_folder_path, 'pytorch_model.bin')
tokenizer_file_path = os.path.join(new_folder_path, 'tokenizer_config.json')

# Split the model files into chunks
split_file(model_file_path)
split_file(tokenizer_file_path)

print('model files splitted.')

###

def assemble_file(file_path, chunk_count):
    with open(file_path, 'wb') as output_file:
        for i in range(chunk_count):
            chunk_file_path = f"{file_path}.part{i}"
            with open(chunk_file_path, 'rb') as chunk_file:
                output_file.write(chunk_file.read())

# Number of chunks for each file
model_chunk_count = len([name for name in os.listdir(new_folder_path) if 'pytorch_model.bin.part' in name])
tokenizer_chunk_count = len([name for name in os.listdir(new_folder_path) if 'tokenizer_config.json.part' in name])

# Define the paths to your model files
assembled_model_file_path = os.path.join(new_folder_path, 'pytorch_model.bin')
assembled_tokenizer_file_path = os.path.join(new_folder_path, 'tokenizer_config.json')

# Assemble the model files from chunks
assemble_file(assembled_model_file_path, model_chunk_count)
assemble_file(assembled_tokenizer_file_path, tokenizer_chunk_count)

print('model files assembled.')

###

# # Load the assembled model
# model = AutoModelForSequenceClassification.from_pretrained(assembled_tokenizer_file_path)

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(assembled_model_file_path)

# # Example input text
# input_text = "Example input text for the model."

# # Tokenize the input text
# inputs = tokenizer(input_text, return_tensors="pt")

# # Run the model to get predictions
# outputs = model(**inputs)

# # Process the output (logits) to get the predicted class
# predictions = outputs.logits.argmax(dim=-1)
# print(f"Predicted class: {predictions.item()}")

###

## Remove files
os.remove(assembled_model_file_path)
os.remove(assembled_tokenizer_file_path)