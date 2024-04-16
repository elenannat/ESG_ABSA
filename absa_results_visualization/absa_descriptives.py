'''Visualization of absa results for paper/presentation'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn for better aesthetics
sns.set(style="whitegrid")

# Import data

# Results
sents = pd.read_excel(r'data/Fewshot_results_timeseries.xlsx')

# ... and Metadata
metadata = pd.read_pickle(r'data/230627_cleaned_dataset.pkl')

# Assuming 'sents' is your DataFrame and it contains a column named 'ESG_Subcategory' for the subcategories
# Create a pivot table
pivot_table = sents.pivot_table(index='ESG-Subcategory', columns='year', values='Net_Sentiment', aggfunc='mean')

# Creating the heatmap
plt.figure(figsize=(12, 10))  # Adjust size to ensure all categories are visible
ax = sns.heatmap(pivot_table, annot=False, fmt=".2f", cmap='coolwarm')
ax.set_title('Average Net Sentiment by ESG Subcategory Over Years')
plt.xticks(rotation=45)  # Rotate the years for better visibility
plt.yticks(rotation=0)   # Ensure ESG subcategory names are readable

plt.savefig(r'./results/ESG_Subcategories_Sentiment_Heatmap.pdf', format='pdf', bbox_inches='tight')  # Save as PDF


plt.figure(figsize=(10, 6))
sns.boxplot(x='ESG-Subcategory', y='Net_Sentiment', data=sents)
plt.title('Distribution of Net Sentiment Scores by ESG Subcategory')
plt.xlabel('ESG Subcategory')
plt.ylabel('Net Sentiment')
plt.xticks(rotation=45)
plt.show()

