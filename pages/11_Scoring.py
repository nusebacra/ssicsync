# streamlit_app.py

import streamlit as st
import pandas as pd

# prep validation data
data_validation_filepath = "data_validation.csv"

data_validation = pd.read_csv(data_validation_filepath, dtype = str)

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

st.pyplot(plt) 

