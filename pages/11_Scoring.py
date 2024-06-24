# streamlit_app.py

import streamlit as st
import pandas as pd

# prep validation data
data_validation_filepath = "LLM_Test\data_validation.csv"

data_validation = pd.read_csv(data_validation_filepath, dtype = str)


# load model directly from huggingface
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("nusebacra/ssicsync_section_classifier")
model = TFAutoModelForSequenceClassification.from_pretrained("nusebacra/ssicsync_section_classifier")



# create ssic denormalized fact table
ssic_detailed_def_filepath = "ssic2020-detailed-definitions.xlsx"
ssic_alpha_index_filepath = "ssic2020-alphabetical-index.xlsx"

df_detailed_def = pd.read_excel(ssic_detailed_def_filepath, skiprows=4)

df_alpha_index = pd.read_excel(ssic_alpha_index_filepath, dtype=str, skiprows=5)
df_alpha_index = df_alpha_index.drop(df_alpha_index.columns[2], axis=1).dropna().rename(columns={'SSIC 2020': 'SSIC 2020','SSIC 2020 Alphabetical Index Description': 'Detailed Definitions'})

df_concat = pd.concat([df_detailed_def, df_alpha_index])

####################################################################################################
# select which fact table to train/transform
# - df_detailed_def
# - df_concat       (concat of df_detailed_def and df_alpha_index)
df_data_dict = df_detailed_def 

# select ssic level of train/test
# - 'Section'
# - 'Division'
# - 'Group'
# - 'Class'
# - 'Subclass'
level = 'Section' 
####################################################################################################

# prep ssic_n tables for joining/merging and reference
# Section, 1-alpha 
ssic_1_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 1)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code']) 
ssic_1_raw['Groups Classified Under this Code'] = ssic_1_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_1 = ssic_1_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_1['Groups Classified Under this Code'] = ssic_1['Groups Classified Under this Code'].str.replace('•', '')
ssic_1['Section, 2 digit code'] = ssic_1['Groups Classified Under this Code'].str[0:2]
ssic_1 = ssic_1.rename(columns={'SSIC 2020': 'Section','SSIC 2020 Title': 'Section Title'})

# Division, 2-digit
ssic_2_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 2)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_2_raw['Groups Classified Under this Code'] = ssic_2_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_2 = ssic_2_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_2['Groups Classified Under this Code'] = ssic_2['Groups Classified Under this Code'].str.replace('•', '')
ssic_2 = ssic_2.rename(columns={'SSIC 2020': 'Division','SSIC 2020 Title': 'Division Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Group, 3-digit 
ssic_3_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 3)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_3_raw['Groups Classified Under this Code'] = ssic_3_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_3 = ssic_3_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_3['Groups Classified Under this Code'] = ssic_3['Groups Classified Under this Code'].str.replace('•', '')
ssic_3 = ssic_3.rename(columns={'SSIC 2020': 'Group','SSIC 2020 Title': 'Group Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Class, 4-digit
ssic_4_raw = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 4)].reset_index(drop=True).drop(columns=['Detailed Definitions', 'Cross References', 'Examples of Activities Classified Under this Code'])
ssic_4_raw['Groups Classified Under this Code'] = ssic_4_raw['Groups Classified Under this Code'].str.split('\n•')
ssic_4 = ssic_4_raw.explode('Groups Classified Under this Code').reset_index(drop=True)
ssic_4['Groups Classified Under this Code'] = ssic_4['Groups Classified Under this Code'].str.replace('•', '')
ssic_4 = ssic_4.rename(columns={'SSIC 2020': 'Class','SSIC 2020 Title': 'Class Title'}).drop(columns=['Groups Classified Under this Code']).drop_duplicates()

# Sub-class, 5-digit
ssic_5 = df_data_dict[df_data_dict['SSIC 2020'].apply(lambda x: len(str(x)) == 5)].reset_index(drop=True).drop(columns=['Groups Classified Under this Code'])
ssic_5.replace('<Blank>', '', inplace=True)
ssic_5.replace('NaN', '', inplace=True)

# prep join columns
ssic_5['Section, 2 digit code'] = ssic_5['SSIC 2020'].astype(str).str[:2]
ssic_5['Division'] = ssic_5['SSIC 2020'].astype(str).str[:2]
ssic_5['Group'] = ssic_5['SSIC 2020'].astype(str).str[:3]
ssic_5['Class'] = ssic_5['SSIC 2020'].astype(str).str[:4]

# join ssic_n Hierarhical Layer Tables (Section, Division, Group, Class, Sub-Class)
ssic_df = pd.merge(ssic_5, ssic_1[['Section', 'Section Title', 'Section, 2 digit code']], on='Section, 2 digit code', how='left')
ssic_df = pd.merge(ssic_df, ssic_2[['Division', 'Division Title']], on='Division', how='left')
ssic_df = pd.merge(ssic_df, ssic_3[['Group', 'Group Title']], on='Group', how='left')
ssic_df = pd.merge(ssic_df, ssic_4[['Class', 'Class Title']], on='Class', how='left')

####################################################################################################
# mapping
level_map = {
    'Section': ('Section', ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)),
    'Division': ('Division', ssic_df.iloc[:, [0, 1, 6, 10, 11, 12, 13]].drop_duplicates(), ssic_2.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)),
    'Group': ('Group', ssic_df.iloc[:, [0, 1, 7, 10, 11, 12, 13]].drop_duplicates(), ssic_3.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)),
    'Class': ('Class', ssic_df.iloc[:, [0, 1, 8, 10, 11, 12, 13]].drop_duplicates(), ssic_4.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)),
    'Subclass': ('Subclass', ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates(), ssic_5.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True))
}

# Get the values for a and b based on the lvl_train
lvl_train, df_streamlit, ssic_n_sl = level_map.get(level, ('default_a', 'default_b', 'default_c'))

lvl_train_title = lvl_train + " Title"

# prep ssic_n dictionary df_prep
df_prep = ssic_df[[lvl_train, 'Detailed Definitions']]
df_prep['encoded_cat'] = df_prep[lvl_train].astype('category').cat.codes
df_prep = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()

# Reference Table for all SSIC Layers
ref_df = df_detailed_def[['SSIC 2020','SSIC 2020 Title']]
ref_df.drop_duplicates(inplace=True)

###############################################################################################################################################
# Indicate list of Top N predictions for scoring 
lvl_train = 'Section'
top_n_predictions = 15
###############################################################################################################################################

# Define the function to predict scores and categories
def predict_text(text, top_n=top_n_predictions):
    predict_input = tokenizer.encode(
        text,
        truncation=True,
        padding=True,
        return_tensors="tf"
    )
    output = tokenizer(predict_input)[0]
    output_array = output.numpy()[0]  # Get the first (and only) output for this input
    # output_array = tf.nn.softmax(output, axis=-1).numpy()[0] # Probability (0-1)

    output_array = output_array   
    
    # Get the top n scores and their corresponding categories
    top_n_indices = output_array.argsort()[-top_n:][::-1]
    top_n_scores = output_array[top_n_indices]
    top_n_categories = top_n_indices
    
    return [{'value': score, 'encoded_cat': category} for score, category in zip(top_n_scores, top_n_categories)]

# Create an empty list to store the predictions
predictions = []

# Iterate over each row of the DataFrame and apply the prediction function
for idx, row in data_validation.iterrows():
    text = row['Notes Page Content2']
    result = predict_text(text)
    for pred in result:
        pred.update({
            'UEN': row['UEN'],
            'entity_name': row['entity_name'],
            'ssic_code': row['ssic_code'],
            'ssic_code2': row['ssic_code2'],
            'Notes Page Content': row['Notes Page Content'],
            'Notes Page Content2': text,
            'Layer': row[lvl_train],
            'Layer2': row[lvl_train + "2"]
        })
        predictions.append(pred)

# Create a DataFrame from the list of predictions
prediction_df = pd.DataFrame(predictions)
prediction_df = pd.merge(prediction_df, ref_df[['SSIC 2020','SSIC 2020 Title']].rename(columns={'SSIC 2020 Title': 'SSIC 2020 Title1'}), left_on='Layer', right_on='SSIC 2020', how='left')
prediction_df.drop(columns=['SSIC 2020'], inplace=True)
prediction_df = pd.merge(prediction_df, ref_df[['SSIC 2020','SSIC 2020 Title']].rename(columns={'SSIC 2020 Title': 'SSIC 2020 Title2'}), left_on='Layer2', right_on='SSIC 2020', how='left')
prediction_df.drop(columns=['SSIC 2020'], inplace=True)
prediction_df = prediction_df.merge(df_prep, on = 'encoded_cat', how = 'left')
prediction_df = prediction_df.merge(ref_df, left_on = 'Section', right_on = 'SSIC 2020', how = 'left')
prediction_df.drop(columns=['SSIC 2020'], inplace=True)
prediction_df = prediction_df.where(pd.notna(prediction_df), "Blank")


grouped_prediction_df = prediction_df.groupby(['entity_name', 'ssic_code', 'ssic_code2', 'Layer', 'Layer2'])[lvl_train].apply(list).reset_index()

def check_alpha_in_list(row, N):
    # Check if the alpha in the 1st column is in the first N elements of the list in the 3rd column
    if row['Layer'] in row['Section'][:N] or row['Layer2'] in row['Section'][:N]:
        return 'Y'
    else:
        return 'N'

###############################################################################################################################################
# Specify Top N Threshold
N = 3
###############################################################################################################################################

grouped_prediction_df['Within Top N'] = grouped_prediction_df.apply(check_alpha_in_list, axis=1, N=N)

# Rank the predictions within each UEN group
prediction_df['Rank'] = prediction_df.groupby('entity_name').cumcount() + 1

def calculate_score(row, top_n_predictions):
    if row['Rank'] <= top_n_predictions:
        if row['Layer'] == row[lvl_train]:
            rank = row['Rank']
            if rank == 1:
                return 0
            else:
                return round((rank - 1) / top_n_predictions,2)
    return 1

def calculate_score2(row, top_n_predictions):
    if pd.isnull(row['Layer2']) or row['Layer2'] == 'Blank' :
        return None
    
    if row['Rank'] <= top_n_predictions:
        if row['Layer2'] == row[lvl_train]:
            rank = row['Rank']
            if rank == 1:
                return 0
            else:
                return round((rank - 1) / top_n_predictions,2)
    return 1

###############################################################################################################################################
# Exponential Scoring
def ecalculate_score(row, top_n_predictions):
    if row['Rank'] <= top_n_predictions and row['Layer'] == row[lvl_train]:
        rank = row['Rank']
        if rank == 1:
            return 0
        else:
            # Exponential transformation
            score = 1 - (1 - 0.1) ** (rank - 1)
            # score = (rank - 1) ** 2 / (top_n_predictions - 1) ** 2
            return round(score, 2)
    return 1

def ecalculate_score2(row, top_n_predictions):
    if pd.isnull(row['Layer2']) or row['Layer2'] == 'Blank':
        return None
    
    if row['Rank'] <= top_n_predictions and row['Layer2'] == row[lvl_train]:
        rank = row['Rank']
        if rank == 1:
            return 0
        else:
            # Exponential transformation
            score = 1 - (1 - 0.1) ** (rank - 1)
            # score = (rank - 1) ** 2 / (top_n_predictions - 1) ** 2
            return round(score, 2)
    return 1


prediction_df['score'] = prediction_df.apply(ecalculate_score, axis=1, args=(top_n_predictions,)) # Toggle ecal or cal
prediction_df['score2'] = prediction_df.apply(ecalculate_score2, axis=1, args=(top_n_predictions,)) # Toggle ecal or cal

# prediction_df

# Find the minimum score for each UEN
score_prediction_df = prediction_df.groupby(['entity_name']).agg({'score': 'min', 'score2': 'min'}).reset_index()

###############################################################################################################################################
# Specify Weightage for Primary and Secondary SSIC
p_weight = 1
s_weight = 0.5

score_prediction_df['t_score'] = round((score_prediction_df['score'] * p_weight + score_prediction_df['score2'].fillna(0) * s_weight) / (p_weight + (score_prediction_df['score2'].notnull() * s_weight)),2)
###############################################################################################################################################

# prediction_df = pd.merge(prediction_df, ref_df[['SSIC 2020','SSIC 2020 Title']].rename(columns={'SSIC 2020 Title': 'SSIC 2020 Title1'}), left_on='Layer', right_on='SSIC 2020', how='left')
score_prediction_df = pd.merge(score_prediction_df, grouped_prediction_df[['entity_name', 'Within Top N']], left_on='entity_name', right_on='entity_name', how='left')

####################################################################################################
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the distribution of t_score colored by category
plt.figure(figsize=(10, 6))  # Optional: Adjust figure size

# Histogram plot with Seaborn
sns.histplot(data=score_prediction_df, x='t_score', hue='Within Top N', multiple='stack', bins=20, edgecolor='black')  # Adjust bins as needed
plt.title('Distribution of t_score by Within Top N, where N = ' + str(N))  # Optional: Add plot title
plt.xlabel('t_score')  # Optional: Add x-axis label
plt.ylabel('Frequency')  # Optional: Add y-axis label

plt.grid(True)  # Optional: Add grid
plt.show()

st.pyplot()