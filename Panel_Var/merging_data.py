'''Merge all data: Results from Few-Shot S&P-ESG-Scores and Refinitiv ESG-Scores'''

import pandas as pd
import re
from unidecode import unidecode
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz import process, fuzz


# Load data from an Excel file
df1 = pd.read_excel(r'./Data/Fewshot_results_timeseries.xlsx')

# Load data from a Stata file (.dta)
df2 = pd.read_stata(r'./Data/SP_ESG/SPData.dta')

# Create cleaning function to prep data for merging
def clean_name(name):
    name = name.lower()
    name = unidecode(name)  # Normalize accents
    # Replace complex legal forms with simplified abbreviations
    name = re.sub(r'\b(aktiengesellschaft)\b', 'ag', name)
    name = re.sub(r'\b(societe en commandite par actions|société en commandite par actions)\b', 'sca', name)
    name = re.sub(r'\b(societe anonyme|société anonyme)\b', 'sa', name)
    name = re.sub(r'[^a-z0-9 ]', '', name)  # Remove non-alphanumeric characters
    name = re.sub(r'\s+', ' ', name).strip()  # Collapse multiple spaces
    # Return only the first token of the cleaned name
    return name.split()[0] if name.split() else name

# Example usage on DataFrame
df1['clean_name'] = df1['Company_name'].apply(clean_name)
df2['clean_name'] = df2['companyname'].apply(clean_name)


# Perform direct matching using Pandas merge for efficiency
matched_df = pd.merge(df1, df2, on='clean_name', how='left', suffixes=('_df1', '_df2'))

# Identify unmatched entries
unmatched_df1 = matched_df[matched_df['Company_name_df2'].isna()]

