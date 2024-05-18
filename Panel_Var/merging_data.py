'''Merge all data: Results from Few-Shot S&P-ESG-Scores, Refinitiv ESG-Scores and Bloomberg'''

import pandas as pd
import re
from unidecode import unidecode

# Load data from an Excel file
df1 = pd.read_excel(r'./Data/Fewshot_results_timeseries_update.xlsx')

# Load data from a Stata file (.dta)
df2 = pd.read_stata(r'./Data/SP_ESG/SPData.dta')

# Load integrated data from stata file

df_int = pd.read_stata(r'./Data/All_ESG/EuroStoxx50_2014-2024.dta')
df_int_annual = pd.read_stata(r'./Data/All_ESG/EuroStoxx50_2014-2024_annual.dta')
df_int_meta = pd.read_excel(r'./Data/All_ESG/EuroStoxx50-ticker-ciq-mapping.xlsx')
df_int_meta_ticker = pd.read_stata(r'./Data/All_ESG/EuroStoxx50CompList.dta')

# Merge the ticker column from df_int_meta onto df_int based on the isin column
df_int_meta_merged = pd.merge(df_int, df_int_meta_ticker[['isin', 'ticker']], on='isin', how='left')


# Create cleaning function to prep data for merging

df2 = df2[['scoredate','csaindustrygroupname','csascoretypename','dimensionname','scorevalue','companyname','country']]


def clean_name(name):
    name = name.lower()
    name = unidecode(name)  # Normalize accents
    # Simplify common legal terms
    name = re.sub(r'\b(aktiengesellschaft)\b', 'ag', name)
    name = re.sub(r'\b(societe en commandite par actions|société en commandite par actions)\b', 'sca', name)
    name = re.sub(r'\b(societe anonyme|société anonyme)\b', 'sa', name)
    name = re.sub(r'[^a-z0-9 ]', '', name)  # Remove non-alphanumeric characters
    name = re.sub(r'\s+', ' ', name).strip()  # Collapse multiple spaces
    tokens = name.split()
    return ' '.join(tokens[:2])  # Return the first two tokens


# Example usage on DataFrame
df1['clean_name'] = df1['Company_name'].apply(clean_name)
df2['clean_name'] = df2['companyname'].apply(clean_name)


# Perform direct matching using Pandas merge for efficiency
matched_df = pd.merge(df1, df2, on='clean_name', how='left', suffixes=('_df1', '_df2'))

# Identify unmatched entries
unmatched_df1 = matched_df[matched_df['Company_name'].isna()]

matched_df['Combined_Names'] = matched_df['Company_name'] + " | " + matched_df['companyname']

matched_df = matched_df[['clean_name','scoredate','csaindustrygroupname','csascoretypename','dimensionname','scorevalue','companyname','country','Combined_Names']]

matched_df['scoredate'] = [date.year for date in matched_df['scoredate']]

mean_scores = matched_df.groupby(['clean_name', 'scoredate', 'country','csaindustrygroupname','csascoretypename','dimensionname'])['scorevalue'].mean().reset_index()

mean_scores.drop_duplicates(inplace=True)

# Save results to pickle
mean_scores.to_pickle(r'./Data/matched_df_update.pkl')
