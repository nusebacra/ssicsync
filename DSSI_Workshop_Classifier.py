import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns


# Set page config
apptitle = 'DSSI Workshop - SSIC Section Classification'

st.set_page_config(page_title=apptitle, layout='wide')

# st.title('SSIC Dictionary')
st.write('Reference: https://docs.streamlit.io/en/stable/api.html#display-data')

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
st.balloons() 
####################

# Load model and test

from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd

###############################################################################################################################################
# Specify directory with model
save_directory = "LLM_Test/distilBert Text Multiclass by 21 Section caa 200624"
###############################################################################################################################################

loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory) #TODO
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory) #TODO



###############################################################################################################################

ssic_detailed_def_filename = "ssic2020-detailed-definitions.xlsx"
ssic_alpha_index_filename = "ssic2020-alphabetical-index.xlsx"

df_detailed_def = pd.read_excel(ssic_detailed_def_filename, skiprows=4)
df_alpha_index = pd.read_excel(ssic_alpha_index_filename, dtype=str, skiprows=5)

df_alpha_index = df_alpha_index.drop(df_alpha_index.columns[2], axis=1).dropna().rename(columns={'SSIC 2020': 'SSIC 2020','SSIC 2020 Alphabetical Index Description': 'Detailed Definitions'})
df_concat = pd.concat([df_detailed_def, df_alpha_index])

###############################################################################################################################################
# Select which dictionary to train
# 1 - df_detailed_def
# 2 - df_concat (df_detailed_def and df_alpha_index)
df_data_dict = df_detailed_def 
###############################################################################################################################################

# Prep SSIC ref-join tables
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

df_streamlit = ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates()
ssic_1_sl = ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)

lvl_train = 'Section'
lvl_train_title = lvl_train + " Title"

df_prep = ssic_df[[lvl_train, 'Detailed Definitions']]
df_prep['encoded_cat'] = df_prep[lvl_train].astype('category').cat.codes

data_texts = df_prep['Detailed Definitions'].to_list() # Features (not tokenized yet)
data_labels = df_prep['encoded_cat'].to_list() # Labels

df_prep = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()


# page title
st.title("Business Description Classifier")
# page subheader
st.subheader("Classify Business Descriptions into 21 Section Categories")

# Add some text explaining the app
st.write("""
Welcome to the Business Description Classifier! This application utilizes a multiclass text classification model 
to categorize business descriptions into one of 21 section categories. Simply input your business description, 
and the model will analyze the text and provide a list predicted categories.

##### How to Use
1. Enter the business description in the text box below.
2. Hit Control + Enter.
3. The top 5 predicted categories will be displayed below the button.

##### About the Model
This model has been trained on a diverse dataset of business descriptions and is capable of understanding and 
classifying a wide range of business activities. The 21 section categories cover various industry sectors, 
providing accurate and meaningful classifications for your business needs.

##### Examples
- **Technology:** Software development, IT consulting, hardware manufacturing.
- **Healthcare:** Hospitals, pharmaceutical companies, medical research.
- **Finance:** Banking, insurance, investment services.

We hope you find this tool helpful for your business classification tasks. If you have any feedback or suggestions, 
please feel free to reach out.
""")

# User input for text description
user_input = st.text_area("Enter Business Description:", "")

if user_input:
    # Process the input text using the model
    predict_input = loaded_tokenizer.encode(user_input, truncation=True, padding=True, return_tensors="tf")
    output = loaded_model(predict_input)[0]
    
    # output_array = tf.nn.softmax(output, axis=-1).numpy()
    output_array = output.numpy() # Logits (+ve to -ve)   
    
    # Define class weights
    class_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    weighted_output_array = output_array * class_weights

    # Create a DataFrame from the output array
    sorted_output_df = pd.DataFrame(weighted_output_array.T, columns=['Score']).sort_values(by='Score', ascending=False)
    sorted_output_df.reset_index(inplace=True)
    sorted_output_df.columns = ['encoded_cat', 'Value']

    lvl_train = 'Section'

    # Conditional statements based on lvl_train
    if lvl_train == 'Section':
        ssic_lvl = ssic_1
    elif lvl_train == 'Division':
        ssic_lvl = ssic_2
    elif lvl_train == 'Group':
        ssic_lvl = ssic_3
    elif lvl_train == 'Class':
        ssic_lvl = ssic_4
    elif lvl_train == 'SSIC 2020':
        ssic_lvl = ssic_5

    # Merge DataFrames
    lvl_dict = df_prep[[lvl_train, 'encoded_cat']].drop_duplicates()
    lvl_ref = ssic_lvl[[lvl_train, lvl_train_title]].drop_duplicates()
    merged_df = lvl_dict.merge(lvl_ref, on='Section', how='left')
    merged_df2 = sorted_output_df.merge(merged_df, on='encoded_cat', how='left')

    # Display the result as a table
    st.subheader("Prediction Results")
    st.table(merged_df2[['Value', lvl_train, lvl_train_title]].head(5))


###############################################################################################################################

# col1, col2, col3 = st.columns([1,0.3,2])

# with col1:
#     # use st.table to display full table w/o scrolling
#     st.table(ssic_1_sl) 

# with col2:
#     section_filter = st.text_input('Search by Section:', '')
#     ssic_filter = st.text_input('Search by SSIC:', '')
#     ssic_2020_title_filter = st.text_input('Search by Title Keywords:', '')

#     # Filtering logic based on user input
#     if section_filter:
#         filtered_df_section = df_streamlit[df_streamlit['Section'].str.contains(section_filter, case=False)]
#     else:
#         filtered_df_section = df_streamlit

#     if ssic_filter:
#         filtered_df_ssic = filtered_df_section[filtered_df_section['SSIC 2020'].str.contains(ssic_filter, case=False)]
#     else:
#         filtered_df_ssic = filtered_df_section

#     if ssic_2020_title_filter:
#         filtered_df_ssic_2020_title = filtered_df_ssic[filtered_df_ssic['SSIC 2020 Title'].str.contains(ssic_2020_title_filter, case=False)]
#     else:
#         filtered_df_ssic_2020_title = filtered_df_ssic
        
# with col3:
#     # st.write(filtered_df_ssic_2020_title, use_container_width=True)
#     st.table(filtered_df_ssic_2020_title)