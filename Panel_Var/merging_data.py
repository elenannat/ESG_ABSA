import pandas as pd
import re
from unidecode import unidecode


def clean_name(name):
    """
    Clean company names for merging by normalizing, removing non-alphanumeric characters,
    and simplifying common legal terms.
    """
    name = name.lower()
    name = unidecode(name)  # Normalize accents
    name = re.sub(r'\b(aktiengesellschaft)\b', 'ag', name)
    name = re.sub(r'\b(societe en commandite par actions|société en commandite par actions)\b', 'sca', name)
    name = re.sub(r'\b(societe anonyme|société anonyme)\b', 'sa', name)
    name = re.sub(r'[^a-z0-9 ]', '', name)  # Remove non-alphanumeric characters
    name = re.sub(r'\s+', ' ', name).strip()  # Collapse multiple spaces
    tokens = name.split()
    return ' '.join(tokens[:2])  # Return the first two tokens


# Load data
df1_old = pd.read_excel('./Data/Fewshot_results_timeseries_update.xlsx')
df1 = pd.read_excel('./Data/Fewshot_results_timeseries_update2.xlsx')
df2 = pd.read_stata('./Data/SP_ESG/SPData.dta')
df_int = pd.read_stata('./Data/All_ESG/EuroStoxx50_2014-2024.dta')
df_int_annual = pd.read_stata('./Data/All_ESG/EuroStoxx50_2014-2024_annual.dta')
df_int_meta = pd.read_excel('./Data/All_ESG/EuroStoxx50-ticker-ciq-mapping.xlsx')
df_int_meta_ticker = pd.read_stata('./Data/All_ESG/EuroStoxx50CompList.dta')
df_int_meta_name = pd.read_csv(
    './Data/All_ESG/Refinitiv-ESG_Scores-2008-2024_ret2024-04-29.csv', sep=';'
)

# Merge names on integrated dataframe
df_int_merged = pd.merge(
    df_int, df_int_meta_name[['ISIN', 'Company Common Name']],
    left_on='isin', right_on='ISIN', how='left'
).drop(columns=['ISIN'])

# Clean names in data frames
df1['clean_name'] = df1['Company_name'].apply(clean_name)
df2['clean_name'] = df2['companyname'].apply(clean_name)

# Dictionary for replacement
replace_dict = {
    'mercedesbenz group': 'mercedes benz',
    'munchener ruckversicherungsgesellschaft': 'muenchener rueckversicherungs',
    'sanofi': 'sanofi sa'
}

# Replace values in the 'companyname' column
df2['clean_name'] = df2['clean_name'].replace(replace_dict)
df_int_merged['clean_name'] = df_int_merged['Company Common Name'].apply(clean_name)

# Perform direct matching using Pandas merge for efficiency
matched_df = pd.merge(df1, df2, on='clean_name', how='left', suffixes=('_df1', '_df2'))

# Identify unmatched entries
unmatched_df1 = matched_df[matched_df['Company_name'].isna()]

matched_df['Combined_Names'] = matched_df['Company_name'] + " | " + matched_df['companyname']
matched_df = matched_df[[
    'clean_name', 'scoredate', 'csaindustrygroupname', 'csascoretypename',
    'dimensionname', 'scorevalue', 'companyname', 'country', 'Combined_Names'
]]
matched_df['scoredate'] = matched_df['scoredate'].dt.year

mean_scores = matched_df.groupby([
    'clean_name', 'scoredate', 'csaindustrygroupname', 'csascoretypename', 'dimensionname'
])['scorevalue'].mean().reset_index()

mean_scores.drop_duplicates(inplace=True)
mean_scores.to_pickle('./Data/matched_df_update3.pkl')

# Create a DataFrame with the list of companies we need
df_companies = pd.DataFrame(matched_df['clean_name'].unique(), columns=['Company Name'])
df_companies.to_excel('./Data/companies.xlsx', index=False)

# Create the Bloomberg scores DataFrame
bloomberg_df = df_int_merged[[
    'Company Common Name', 'clean_name', 'scoredate',
    'escore_bloomberg', 'sscore_bloomberg', 'gscore_bloomberg'
]].dropna()

# Create the S&P scores DataFrame
sp_df = df_int_merged[[
    'Company Common Name', 'clean_name', 'scoredate',
    'escore_sp', 'sscore_sp', 'gscore_sp'
]].dropna()

# Create the Refinitiv scores DataFrame
refinitiv_df = df_int_merged[[
    'Company Common Name', 'clean_name', 'scoredate',
    'escore_ref', 'sscore_ref', 'gscore_ref'
]].dropna()

# Group by year and name
bloomberg_df['scoredate'] = pd.to_datetime(bloomberg_df['scoredate'])
bloomberg_df['year'] = bloomberg_df['scoredate'].dt.year
bloomberg_grouped = bloomberg_df.groupby([
    'Company Common Name', 'clean_name', 'year'
]).mean().reset_index()

print(bloomberg_grouped['clean_name'].unique())
