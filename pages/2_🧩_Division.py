import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

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

df_1_streamlit = ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates()
df_2_streamlit = ssic_df.iloc[:, [0, 1, 6, 10, 11, 12, 13]].drop_duplicates()
df_3_streamlit = ssic_df.iloc[:, [0, 1, 7, 10, 11, 12, 13]].drop_duplicates()
df_4_streamlit = ssic_df.iloc[:, [0, 1, 8, 10, 11, 12, 13]].drop_duplicates()
df_5_streamlit = ssic_df.iloc[:, [0, 1, 9, 10, 11, 12, 13]].drop_duplicates()

ssic_1_sl = ssic_1.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)
ssic_2_sl = ssic_2.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)
ssic_3_sl = ssic_3.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)
ssic_4_sl = ssic_4.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)
ssic_5_sl = ssic_5.iloc[:, [0, 1]].drop_duplicates().reset_index(drop=True)

df_streamlit = df_2_streamlit
ssic_sl = ssic_2_sl

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
# Streamlit Page UI Config

import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(
    page_title='ssicsync', # Set display name of browser tab
    page_icon="🔍", # Set display icon of browser tab
    layout="wide", # "wide" or "centered"
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''Explore multiclass text classification with DistilBERT on our Streamlit page. 
        Discover interactive insights and the power of modern NLP in text categorization!'''
    }
)

# st.title('SSIC Dictionary')
st.write('Reference: https://docs.streamlit.io/en/stable/api.html#display-data')

# Visual Effects ### - https://docs.streamlit.io/develop/api-reference/status
st.balloons() 

# Guide to Streamlit Text Elements - https://docs.streamlit.io/develop/api-reference/text

# Define CSS styles
custom_styles = """
<style>
    .appview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }


</style>
"""
    # img.full-width {
    #     max-width: 100%;
    #     width: 100vw; /* Set image width to full viewport width */
    #     height: auto; /* Maintain aspect ratio */
    #     display: block; /* Remove any default space around the image */
    #     margin-left: auto;
    #     margin-right: auto;
    # }

# Display CSS styles using st.markdown
st.markdown(custom_styles, unsafe_allow_html=True)

st.header('🧩 Division, 81 Categories', divider='rainbow')

col1, col2, col3, col4, col5 = st.columns([3,1,1,1.5,3.5])

with col1:
    st.markdown('''
    <br><br>
    Division Reference Table
    ''', unsafe_allow_html=True)

with col2:
    section_filter = st.text_input('Search by Division:', '')

with col3:
    ssic_filter = st.text_input('Search by SSIC:', '')

with col4:
    ssic_2020_title_filter = st.text_input('Search by Title Keywords:', '')

    # Filtering logic based on user input
    if section_filter:
        filtered_df_ref = ssic_sl[ssic_sl['Division'].str.contains(section_filter, case=False)]
    else:
        filtered_df_ref = ssic_sl

    if section_filter:
        filtered_df_section = df_streamlit[df_streamlit['Division'].str.contains(section_filter, case=False)]
    else:
        filtered_df_section = df_streamlit

    if ssic_filter:
        filtered_df_ssic = filtered_df_section[filtered_df_section['SSIC 2020'].str.contains(ssic_filter, case=False)]
    else:
        filtered_df_ssic = filtered_df_section

    if ssic_2020_title_filter:
        filtered_df_ssic_2020_title = filtered_df_ssic[filtered_df_ssic['SSIC 2020 Title'].str.contains(ssic_2020_title_filter, case=False)]
    else:
        filtered_df_ssic_2020_title = filtered_df_ssic


col1, col2 = st.columns([2,3])

with col1:
    st.write(filtered_df_ref, use_container_width=True)
    # st.table(ssic_sl) # use st.table to display full table w/o scrolling

       
with col2:
    st.write(filtered_df_ssic_2020_title, use_container_width=True)
    # st.table(filtered_df_ssic_2020_title)