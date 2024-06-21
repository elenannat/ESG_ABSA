'''Visualization of absa results for paper/presentation'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud


# Set the style of seaborn for better aesthetics
sns.set(style="whitegrid")

# Import data

# Results
sents_old = pd.read_excel(r'data/Fewshot_results_timeseries_update.xlsx')
sents = pd.read_excel(r'data/Fewshot_results_timeseries_update2.xlsx')

sents = sents[sents['ESG-Subcategory'] != 'non-ESG']
sents = sents[sents['ESG-Subcategory'] != 'Governance']


# ... and Metadata
metadata_old = pd.read_pickle(r'data/230627_cleaned_dataset.pkl')
metadata = pd.read_pickle(r'data/240610_cleaned_dataset.pkl')

# Assuming 'sents' is your DataFrame and it contains a column named 'ESG_Subcategory' for the subcategories
# Create a pivot table
pivot_table = sents.pivot_table(index='ESG-Subcategory', columns='year', values='Net_Sentiment', aggfunc='mean')

pivot_table.drop(['ESG'], inplace=True)


# Creating the heatmap
plt.figure(figsize=(12, 10))  # Adjust size to ensure all categories are visible
ax = sns.heatmap(pivot_table, annot=False, fmt=".2f", cmap='RdYlGn')  # Turn off annotations

# Enhance the title and axis labels with bold formatting
ax.set_title('Average Net Sentiment by ESG Subcategory Over Years', fontsize=16, weight='bold')

# Increase font size for xticks and yticks and make them bold
plt.xticks(rotation=45, fontsize=12, weight='bold')  # Rotate the years for better visibility and bold font
plt.yticks(rotation=0, fontsize=12, weight='bold')   # Ensure ESG subcategory names are readable and bold

# Optionally increase colorbar label size and make them bold
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.ax.yaxis.label.set_weight('bold')


plt.savefig(r'./results/ESG_Subcategories_Sentiment_Heatmap_update2.pdf', format='pdf', bbox_inches='tight')
plt.savefig(r'./results/ESG_Subcategories_Sentiment_Heatmap_update2.eps', bbox_inches='tight')


plt.figure(figsize=(10, 6))
sns.boxplot(x='ESG-Subcategory', y='Net_Sentiment', data=sents)
plt.title('Distribution of Net Sentiment Scores by ESG Subcategory')
plt.xlabel('ESG Subcategory')
plt.ylabel('Net Sentiment')
plt.xticks(rotation=45)
plt.show()

# Word clouds for all reports

entities = pd.read_excel(r'data/entities_df_update.xlsx').drop(columns=['Unnamed: 0'])

# Calculate net sentiment
entities['Net_Sentiment'] = ((entities['Positive'] - entities['Negative']) /
                             (entities['Positive'] + entities['Negative'] + entities['Neutral']))

# Select the top 100 entities based on total count
top_entities = entities.nlargest(100, 'Total')

# Create a dictionary for the word cloud with the total count as frequency
word_freq = dict(zip(top_entities['Entity'], top_entities['Total']))

# Normalize the net sentiment to be between 0 and 1 for color mapping
norm_sentiment = ((top_entities['Net_Sentiment'] - top_entities['Net_Sentiment'].min()) /
                  (top_entities['Net_Sentiment'].max() - top_entities['Net_Sentiment'].min()))

# Generate colors from green to red
colors = plt.colormaps['RdYlGn'](norm_sentiment)


# Create a color function for the word cloud
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    index = top_entities[top_entities['Entity'] == word].index[0]
    color = colors[index]
    return tuple(int(c * 255) for c in color[:3])


# Generate the word cloud
wordcloud = WordCloud(width=800, height=400,
                      background_color='white', color_func=color_func).generate_from_frequencies(word_freq)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Create the 'results' directory if it doesn't exist
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Save the word cloud to the 'results' folder
output_file = os.path.join(output_dir, "wordcloud2.png")
plt.savefig(output_file, format='png', bbox_inches='tight')


# Save the word cloud to the 'results' folder as an EPS file
output_file = os.path.join(output_dir, "wordcloud2.eps")

# Create a figure with no background and no axis, with higher DPI
fig = plt.figure(figsize=(10, 10), dpi=500)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# Save the figure as an EPS file
plt.savefig(output_file, format='eps', bbox_inches='tight')
plt.close(fig)

# Show the plot
plt.show()

# Calculate the average sentiment for the word cloud

average_sentiment = top_entities['Net_Sentiment'].mean()
print(f"Average Sentiment: {average_sentiment}")

# do the same for just E, S and G

mapping_df = pd.read_excel(r'data/merged_entities_et.xlsx')

# Calculate net sentiment
mapping_df['Net_Sentiment'] = (mapping_df['Positive'] - mapping_df['Negative']) / (
            mapping_df['Positive'] + mapping_df['Negative'] + mapping_df['Neutral'])

# Define the number of top entities to select
top_n = 100


# Define a function to create word clouds for a specific ESG category
def create_wordcloud_for_esg_category(df, category, output_dir, top_n=top_n):
    # Filter the dataframe for the specific ESG category
    category_df = df[df['ESG'] == category].copy()

    # Select the top entities based on total count
    category_df['Total'] = category_df['Positive'] + category_df['Negative'] + category_df['Neutral']
    top_entities = category_df.nlargest(top_n, 'Total')

    # Create a dictionary for the word cloud with the total count as frequency
    word_freq = dict(zip(top_entities['Entity'], top_entities['Total']))

    # Normalize the net sentiment to be between 0 and 1 for color mapping
    net_sentiment_dict = dict(zip(top_entities['Entity'], top_entities['Net_Sentiment']))
    min_sentiment = min(net_sentiment_dict.values())
    max_sentiment = max(net_sentiment_dict.values())
    norm_sentiment_dict = {k: (v - min_sentiment) / (max_sentiment - min_sentiment) for k, v in
                           net_sentiment_dict.items()}

    # Generate colors from green to red using the updated colormap method
    color_map = plt.colormaps['RdYlGn']

    # Create a color function for the word cloud
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        norm_sentiment = norm_sentiment_dict[word]
        color = color_map(norm_sentiment)
        return tuple(int(c * 255) for c in color[:3])

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          color_func=color_func).generate_from_frequencies(word_freq)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save the word cloud to a file
    output_file = os.path.join(output_dir, f"wordcloud_{category}2.png")
    plt.savefig(output_file, format='png', bbox_inches='tight')
    plt.close()

    # Calculate and print the average sentiment for the category
    average_sentiment = top_entities['Net_Sentiment'].mean()
    print(f"Average Sentiment for {category}: {average_sentiment}")


# Create the 'results' directory if it doesn't exist
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Create word clouds for each ESG category
for category in ['E', 'S', 'G']:
    create_wordcloud_for_esg_category(mapping_df, category, output_dir)
