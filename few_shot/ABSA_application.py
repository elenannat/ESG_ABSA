import torch
import json
from tqdm import tqdm
from setfit import AbsaModel
from collections import defaultdict
import pandas as pd
import inflect

# Clear CUDA cache
torch.cuda.empty_cache()

def process_in_batches(model, sentences, batch_size=500):
    """
    Process sentences in batches and aggregate results.
    """
    all_predictions = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start + batch_size]
        predictions = model.predict(batch)
        all_predictions.extend(predictions)
    return all_predictions

# Load trained ABSA model on GPU
model = AbsaModel.from_pretrained(
    "models_ESG/setfit-absa-model-aspect_ESG_update",
    "models_ESG/setfit-absa-model-polarity_ESG_update",
    device=0
)

# Load the dataset to be classified
reports_df = pd.read_pickle('../Cleaning_Texts/240610_cleaned_dataset.pkl')

# Process each report
for report_idx, report_row in tqdm(reports_df.iterrows(), total=reports_df.shape[0], desc="Processing Reports"):
    sentences = report_row["filtered_sentences2"]
    if isinstance(sentences, str):
        sentences = [sentences]

    # Classify sentences in batches and save results as JSON
    predictions = process_in_batches(model, sentences)
    file_path = f'./results_absa/report_results_{report_idx}_update2.json'
    with open(file_path, 'w') as file:
        json.dump(predictions, file)

    # Clear CUDA cache to free up memory
    torch.cuda.empty_cache()
    print(f'Result saved for report {report_idx}')

# Directory where JSON files are saved
results_directory = './results_absa/'
data_list = []

# Load results from JSON files
for file_idx in range(2038):
    filename = f'report_results_{file_idx}_update2.json'
    file_path = f'{results_directory}/{filename}'
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            data_list.append(data)
    except FileNotFoundError:
        print(f'File not found: {filename}')
        continue

print(len(data_list))  # Verify the number of loaded reports

def process_entry(entry):
    """
    Process entries to create a sentiment index from ABSA model results.
    """
    inflect_engine = inflect.engine()
    entity_polarity_counts = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0})

    if all(not sublist for sublist in entry):
        return pd.DataFrame([{'Entity': 'no entities', 'Positive': 0, 'Neutral': 0, 'Negative': 0, 'Total': 0}])

    for sublist in entry:
        for item in sublist:
            entity = item['span'].lower()
            entity_singular = inflect_engine.singular_noun(entity) if inflect_engine.singular_noun(entity) else entity
            polarity = item['polarity']
            entity_polarity_counts[entity_singular][polarity] += 1

    df_data = [{'Entity': entity, 'Positive': counts['positive'], 'Neutral': counts['neutral'], 'Negative': counts['negative']}
               for entity, counts in entity_polarity_counts.items()]
    entry_df = pd.DataFrame(df_data)
    entry_df['Total'] = entry_df['Positive'] + entry_df['Neutral'] + entry_df['Negative']
    return entry_df

# Process all entries
entry_dfs = [process_entry(entry) for entry in data_list]

# Concatenate all DataFrames into one
all_entries_df = pd.concat(entry_dfs, keys=range(len(entry_dfs)))
all_entries_reset_df = all_entries_df.reset_index(level=0).rename(columns={'level_0': 'DataFrame_ID'})

# Group by entity and filter entities appearing in 20 or more reports
entity_report_counts = all_entries_reset_df.groupby('Entity')['DataFrame_ID'].nunique()
entities_in_20_or_more_reports = entity_report_counts[entity_report_counts >= 20].index
filtered_entries_df = all_entries_reset_df[all_entries_reset_df['Entity'].isin(entities_in_20_or_more_reports)]
filtered_entries_df = filtered_entries_df.drop('DataFrame_ID', axis=1)

# Aggregate sentiment counts
aggregated_entities_df = filtered_entries_df.groupby('Entity').agg({
    'Positive': 'sum',
    'Neutral': 'sum',
    'Negative': 'sum',
    'Total': 'sum'
}).reset_index()

aggregated_entities_df.sort_values(by='Total', ascending=False, inplace=True)

# Filter entities mentioned more than 100 times and save results
entities_mentioned_more_than_100 = aggregated_entities_df[aggregated_entities_df['Total'] >= 100]
entities_mentioned_more_than_100.to_excel(r'./results/entities_df_update.xlsx')

# Load and process ESG mapping data
esg_mapping_df = pd.read_excel(r'mapping_data/merged_entities_et.xlsx')
esg_mapping_df.dropna(inplace=True)
esg_mapping = esg_mapping_df[['Entity', 'ESG-Subcategory']].drop_duplicates()

# Merge ESG mapping with each DataFrame
for idx, entry_df in enumerate(entry_dfs):
    entry_dfs[idx] = pd.merge(entry_df, esg_mapping, on='Entity', how='left')

def calculate_esg_metrics(input_df):
    """
    Calculate ESG metrics including net sentiment, ESG share, and quantity.
    """
    df_copy = input_df.copy()
    df_copy['ESG-Subcategory'] = df_copy['ESG-Subcategory'].fillna('non-ESG')
    grouped_df = df_copy.groupby('ESG-Subcategory').agg(
        Positive=('Positive', 'sum'),
        Neutral=('Neutral', 'sum'),
        Negative=('Negative', 'sum'),
        Total=('Total', 'sum')
    ).reset_index()

    grouped_df['Net Sentiment'] = (grouped_df['Positive'] - grouped_df['Negative']) / (grouped_df['Positive'] + grouped_df['Negative'] + grouped_df['Neutral'])
    total_count_esg_only = grouped_df[grouped_df['ESG-Subcategory'] != 'non-ESG']['Total'].sum()
    grouped_df['ESG-Share'] = grouped_df.apply(lambda x: x['Total'] / total_count_esg_only if x['ESG-Subcategory'] != 'non-ESG' else 0, axis=1)
    total_esg_and_non_esg = grouped_df['Total'].sum()
    grouped_df['Quantity'] = grouped_df['Total'] / total_esg_and_non_esg

    return grouped_df

# Calculate ESG metrics for each DataFrame
processed_dfs = [calculate_esg_metrics(entry_df) for entry_df in entry_dfs]
reports_df['results'] = processed_dfs

# Convert date column to datetime and extract year
reports_df['date'] = pd.to_datetime(reports_df['date'])
reports_df['year'] = reports_df['date'].dt.year

def aggregate_results(group_df):
    """
    Aggregate results for a given group, calculating net sentiment, ESG share, and quantity.
    """
    concatenated_df = pd.concat(group_df['results'].tolist())
    concatenated_df['Net Sentiment'] = (concatenated_df['Positive'] - concatenated_df['Negative']) / (concatenated_df['Positive'] + concatenated_df['Negative'] + concatenated_df['Neutral'])

    aggregated_df = concatenated_df.groupby('ESG-Subcategory').agg(
        Total=('Total', 'sum'),
        Net_Sentiment=('Net Sentiment', 'mean')
    )

    total_sum = aggregated_df['Total'].sum()
    aggregated_df['ESG-Share'] = aggregated_df['Total'] / total_sum
    aggregated_df['Quantity'] = aggregated_df['Total'] / total_sum

    return aggregated_df

# Group by company name and year, then apply aggregation function
time_series_analysis_df = reports_df.groupby(['Company_name', 'year']).apply(aggregate_results).reset_index()

# Save the final time series analysis results
time_series_analysis_df.to_excel(r'./results/Fewshot_results_timeseries_update2.xlsx')
