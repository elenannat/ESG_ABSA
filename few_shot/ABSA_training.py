import pandas as pd
from datasets import Dataset, concatenate_datasets
from setfit import AbsaTrainer, AbsaModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

# Clear CUDA cache
torch.cuda.empty_cache()


def compute_metrics(y_pred, y_test):
    """
    Compute multiple metrics for evaluating the ABSA model.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets, ensuring unique groups of sentences.
    """
    df['group'] = df.groupby('text').ngroup()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_idx, test_idx in gss.split(df, groups=df['group']):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

    train_df = train_df.drop(columns=['group'])
    test_df = test_df.drop(columns=['group'])

    return train_df, test_df


# Load datasets
env_df = pd.read_excel('train_test_data/env_et_cf_ch_final.xlsx')
gov_df = pd.read_excel('train_test_data/gov_et_cf_ch_final.xlsx')
soc_df = pd.read_excel('train_test_data/soc_et_cf_ch_final.xlsx')

# Split datasets
train_env_df, test_env_df = split_dataset(env_df, test_size=0.20, random_state=420)
train_soc_df, test_soc_df = split_dataset(soc_df, test_size=0.20, random_state=420)
train_gov_df, test_gov_df = split_dataset(gov_df, test_size=0.20, random_state=420)

# Convert pandas DataFrames to Hugging Face Datasets
train_env_dataset = Dataset.from_pandas(train_env_df)
train_soc_dataset = Dataset.from_pandas(train_soc_df)
train_gov_dataset = Dataset.from_pandas(train_gov_df)
test_env_dataset = Dataset.from_pandas(test_env_df)
test_soc_dataset = Dataset.from_pandas(test_soc_df)
test_gov_dataset = Dataset.from_pandas(test_gov_df)

# Concatenate the datasets
concatenated_train_dataset = concatenate_datasets([train_env_dataset, train_soc_dataset, train_gov_dataset])
concatenated_test_dataset = concatenate_datasets([test_env_dataset, test_soc_dataset, test_gov_dataset])

# Load a model from the Hugging Face Hub
model = AbsaModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Create a trainer
trainer = AbsaTrainer(model, train_dataset=concatenated_train_dataset, metric=compute_metrics)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("models_ESG/setfit-absa-model-aspect_ESG_update",
                      "models_ESG/setfit-absa-model-polarity_ESG_update")

# Evaluate the model
metrics = trainer.evaluate(concatenated_test_dataset)
print(metrics)
