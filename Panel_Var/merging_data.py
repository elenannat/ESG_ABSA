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
    name = re.sub(r'\b(aktiengesellschaft)\b', 'ag', name)
    name = re.sub(r'\b(societe en commandite par actions|société en commandite par actions)\b', 'sca', name)
    name = re.sub(r'\b(societe anonyme|société anonyme)\b', 'sa', name)
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def match_item(item, choices, threshold=90):
    match = process.extractOne(item, choices, scorer=fuzz.WRatio, score_cutoff=threshold)
    if match:
        return (item, match[0], match[1])
    return (item, None, None)

def get_matches(df, col1, df2, col2, threshold=90):
    choices = df2[col2].tolist()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(match_item, df[col1], [choices]*len(df), [threshold]*len(df)))
    return pd.DataFrame(results, columns=[col1, 'matched_name', 'score'])

# Use this function as before
matched_df = get_matches(df1, 'clean_name', df2, 'clean_name')

# Apply the cleaning function to the company name columns
df1['clean_name'] = df1['Company_name'].apply(clean_name)
df2['clean_name'] = df2['companyname'].apply(clean_name)

# Get matches with a default threshold of 90
matched_df = get_matches(df1, 'clean_name', df2, 'clean_name')

# Filter unmatched entries
unmatched_df1 = matched_df[matched_df['matched_name'].isnull()]
