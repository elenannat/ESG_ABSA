'''Merge all data: Results from Few-Shot S&P-ESG-Scores and Refinitiv ESG-Scores'''

import pandas as pd
import re
from unidecode import unidecode
from fuzzywuzzy import process


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


def get_matches(df, col1, df2, col2, threshold=90):
    # Store matches in a list
    matches = []

    # Iterate over each item in column 1
    for item in df[col1]:
        # Find best match in df2[col2]
        match = process.extractOne(item, df2[col2], score_cutoff=threshold)
        if match:
            matches.append((item, match[0], match[1]))  # Append the match and score
        else:
            matches.append((item, None, None))  # No match found

    # Return a DataFrame of matches
    return pd.DataFrame(matches, columns=[col1, 'matched_name', 'score'])

# Apply the cleaning function to the company name columns
df1['clean_name'] = df1['Company_name'].apply(clean_name)
df2['clean_name'] = df2['companyname'].apply(clean_name)

# Get matches with a default threshold of 90
matched_df = get_matches(df1, 'clean_name', df2, 'clean_name')

# Filter unmatched entries
unmatched_df1 = matched_df[matched_df['matched_name'].isnull()]
