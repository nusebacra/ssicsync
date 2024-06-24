# import tensorflow as tf
import shutil
import pandas as pd
import os
from datetime import datetime
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification
# from tensorflow.keras.models import load_model

############################################################################################################################################################
# Prep training and reference dataset

# Get current directory
current_dir = os.getcwd()
# Get parent directory
parent_dir = os.path.dirname(current_dir)
# parent_dir = "/SSICSYNC/"

ssic_detailed_def_filename = "ssic2020-detailed-definitions.xlsx"
ssic_alpha_index_filename = "ssic2020-alphabetical-index.xlsx"

# Define the relative path to the CSV file
ssic_detailed_def_filepath = os.path.join(parent_dir, ssic_detailed_def_filename)
ssic_alpha_index_filepath = os.path.join(parent_dir, ssic_alpha_index_filename)

df_detailed_def = pd.read_excel(ssic_detailed_def_filepath, skiprows=4)
df_alpha_index = pd.read_excel(ssic_alpha_index_filepath, dtype=str, skiprows=5)
df_alpha_index = df_alpha_index.drop(df_alpha_index.columns[2], axis=1).dropna().rename(columns={'SSIC 2020': 'SSIC 2020','SSIC 2020 Alphabetical Index Description': 'Detailed Definitions'})

df = pd.concat([df_detailed_def, df_alpha_index])

# Prep SSIC ref-join tables
# Section, 1-alpha 
ssic_1_raw = df[df['SSIC 2020'].apply(lambda x: len(str(x)) == 1)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code']) 
ssic_1_raw['Groups Classified Under this Code'] = ssic_1_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_1 = ssic_1_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_1['Groups Classified Under this Code'] = ssic_1['Groups Classified Under this Code'].str.replace('•', '')
ssic_1['Section, 2 digit code'] = ssic_1['Groups Classified Under this Code'].str[0:2]
ssic_1 = ssic_1.rename(columns={'SSIC 2020': 'Section','SSIC 2020 Title': 'Section Title'})

# Division, 2-digit
ssic_2_raw = df[df['SSIC 2020'].apply(lambda x: len(str(x)) == 2)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_2_raw['Groups Classified Under this Code'] = ssic_2_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_2 = ssic_2_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_2['Groups Classified Under this Code'] = ssic_2['Groups Classified Under this Code'].str.replace('•', '')
ssic_2 = ssic_2.rename(columns={'SSIC 2020': 'Division','SSIC 2020 Title': 'Division Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Group, 3-digit 
ssic_3_raw = df[df['SSIC 2020'].apply(lambda x: len(str(x)) == 3)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_3_raw['Groups Classified Under this Code'] = ssic_3_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_3 = ssic_3_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_3['Groups Classified Under this Code'] = ssic_3['Groups Classified Under this Code'].str.replace('•', '')
ssic_3 = ssic_3.rename(columns={'SSIC 2020': 'Group','SSIC 2020 Title': 'Group Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Class, 4-digit
ssic_4_raw = df[df['SSIC 2020'].apply(lambda x: len(str(x)) == 4)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_4_raw['Groups Classified Under this Code'] = ssic_4_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_4 = ssic_4_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_4['Groups Classified Under this Code'] = ssic_4['Groups Classified Under this Code'].str.replace('•', '')
ssic_4 = ssic_4.rename(columns={'SSIC 2020': 'Class','SSIC 2020 Title': 'Class Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Sub-class, 5-digit
ssic_5 = df[df['SSIC 2020'].apply(lambda x: len(str(x)) == 5)].reset_index(drop=True).drop(columns=['Groups Classified Under this Code'])
ssic_5.replace('<Blank>', '', inplace=True)
ssic_5.replace('NaN', '', inplace=True)

# Prep join columns
ssic_5['Section, 2 digit code'] = ssic_5['SSIC 2020'].astype(str).str[:2]
ssic_5['Division'] = ssic_5['SSIC 2020'].astype(str).str[:2]
ssic_5['Group'] = ssic_5['SSIC 2020'].astype(str).str[:3]
ssic_5['Class'] = ssic_5['SSIC 2020'].astype(str).str[:4]

# Join ssic_5 to Hierarhical Layer Tables (Section, Division, Group, Class, Sub-Class)
ssic_df = pd.merge(ssic_5, ssic_1[['Section', 'Section Title', 'Section, 2 digit code']], on='Section, 2 digit code', how='left')
ssic_df = pd.merge(ssic_df, ssic_2[['Division', 'Division Title']], on='Division', how='left')
ssic_df = pd.merge(ssic_df, ssic_3[['Group', 'Group Title']], on='Group', how='left')
ssic_df = pd.merge(ssic_df, ssic_4[['Class', 'Class Title']], on='Class', how='left')

ref_df = df_detailed_def[['SSIC 2020','SSIC 2020 Title']]
ref_df.drop_duplicates(inplace=True)

df_prep = ssic_df[['Section', 'Detailed Definitions']]
df_prep['encoded_cat'] = df_prep['Section'].astype('category').cat.codes

data_texts = df_prep['Detailed Definitions'].to_list() # Features (not tokenized yet)
data_labels = df_prep['encoded_cat'].to_list() # Labels

df_prep = df_prep[['Section', 'encoded_cat']].drop_duplicates()

############################################################################################################################################################



############################################################################################################################################################
# Classification Model Training
from sklearn.model_selection import train_test_split
 
# Split Train and Validation data
# train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0, shuffle=True)

# 100% of Data for training
train_texts = data_texts
train_labels = data_labels
val_texts = data_texts
val_labels = data_labels
 
# Keep some data for inference (testing)
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=0, shuffle=True)


from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings),val_labels))

# TFTrainer Class for Fine-tuning
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=21)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping
 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
 
model.fit(train_dataset.shuffle(1000).batch(64),
epochs=1,
batch_size=64,
validation_data=val_dataset.shuffle(1000).batch(64)
# ,callbacks=[early_stopping]
)


############################################################################################################################################################

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