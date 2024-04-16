'''Visualization of absa results for paper/presentation'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn for better aesthetics
sns.set(style="whitegrid")

sents = pd.read_excel(r'data/Fewshot_results_timeseries.xlsx')