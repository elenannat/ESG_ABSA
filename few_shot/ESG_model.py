import pandas as pd
import numpy as np
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, KFold
import random
import torch

# Clear CUDA cache
torch.cuda.empty_cache()

# Set the random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)


# Define function to compute multiple metrics for the model evaluation
def compute_metrics(y_pred, y_test):
    """
    Compute multiple metrics for evaluating the model.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


# Define function to split the dataset
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
env_df = pd.read_csv('train_test_data/environmental_2k.csv', index_col=[0]).reset_index(drop=True)
gov_df = pd.read_csv('train_test_data/governance_2k.csv', index_col=[0]).reset_index(drop=True)
soc_df = pd.read_csv('train_test_data/social_2k.csv', index_col=[0]).reset_index(drop=True)

# Rename columns
env_df.rename(columns={'env': 'label'}, inplace=True)
gov_df.rename(columns={'gov': 'label'}, inplace=True)
soc_df.rename(columns={'soc': 'label'}, inplace=True)

# Use a smaller subset for quick experimentation
env_df_small = env_df.iloc[0:200]
gov_df_small = gov_df.iloc[0:200]
soc_df_small = soc_df.iloc[0:200]

# Split datasets
train_env_df, test_env_df = split_dataset(env_df_small, test_size=0.20, random_state=42)
train_soc_df, test_soc_df = split_dataset(soc_df_small, test_size=0.20, random_state=42)
train_gov_df, test_gov_df = split_dataset(gov_df_small, test_size=0.20, random_state=42)

# Convert pandas DataFrames to Hugging Face Datasets
train_env_dataset = Dataset.from_pandas(train_env_df)
train_soc_dataset = Dataset.from_pandas(train_soc_df)
train_gov_dataset = Dataset.from_pandas(train_gov_df)
test_env_dataset = Dataset.from_pandas(test_env_df)
test_soc_dataset = Dataset.from_pandas(test_soc_df)
test_gov_dataset = Dataset.from_pandas(test_gov_df)

# Load untrained models
model_E = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
model_S = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
model_G = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# Training arguments
training_args = TrainingArguments(seed=42)

# Create trainers for each model
trainer_E = Trainer(model_E, train_dataset=train_env_dataset, eval_dataset=test_env_dataset, args=training_args,
                    compute_metrics=compute_metrics)
trainer_S = Trainer(model_S, train_dataset=train_soc_dataset, eval_dataset=test_soc_dataset, args=training_args,
                    compute_metrics=compute_metrics)
trainer_G = Trainer(model_G, train_dataset=train_gov_dataset, eval_dataset=test_gov_dataset, args=training_args,
                    compute_metrics=compute_metrics)

# Train the models
trainer_E.train()
trainer_S.train()
trainer_G.train()

# Save the trained models
model_E.save_pretrained("models_ESG/setfit-paraphrase-mpnet-base-v2-E")
model_S.save_pretrained("models_ESG/setfit-paraphrase-mpnet-base-v2-S")
model_G.save_pretrained("models_ESG/setfit-paraphrase-mpnet-base-v2-G")

# Evaluate the models
metrics_E = trainer_E.evaluate(test_env_dataset)
print("Environmental Model Metrics:", metrics_E)

metrics_S = trainer_S.evaluate(test_soc_dataset)
print("Social Model Metrics:", metrics_S)

metrics_G = trainer_G.evaluate(test_gov_dataset)
print("Governance Model Metrics:", metrics_G)


# Cross-validation function
def cross_validate(model, df, metric, args, n_splits=5, random_seed=42):
    """
    Perform cross-validation on the given model and dataset.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"\nFold {fold + 1}/{n_splits}")

        train_fold_df = df.iloc[train_index]
        test_fold_df = df.iloc[test_index]

        print(f"Training data size: {len(train_fold_df)}")
        print(f"Test data size: {len(test_fold_df)}")
        print(f"Training labels distribution:\n{train_fold_df['label'].value_counts()}")
        print(f"Test labels distribution:\n{test_fold_df['label'].value_counts()}")

        train_dataset = Dataset.from_pandas(train_fold_df)
        test_dataset = Dataset.from_pandas(test_fold_df)

        trainer = Trainer(model=model, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=metric,
                          args=args)

        # Train the model
        print("Training...")
        trainer.train()

        # Evaluate the model
        print("Evaluating...")
        result = trainer.evaluate(test_dataset)
        print(f"Result: {result}")

        # Get predictions
        y_pred = model.predict(test_dataset['text'])
        y_true = test_fold_df['label'].values

        # Print true labels and predictions
        print("True labels:", y_true)
        print("Predictions:", y_pred)

        # Print classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred))

        metrics["accuracy"].append(result["eval_accuracy"])
        metrics["precision"].append(result["eval_precision"])
        metrics["recall"].append(result["eval_recall"])
        metrics["f1"].append(result["eval_f1"])

    # Calculate average metrics
    avg_metrics = {
        "accuracy": np.mean(metrics["accuracy"]),
        "precision": np.mean(metrics["precision"]),
        "recall": np.mean(metrics["recall"]),
        "f1": np.mean(metrics["f1"])
    }

    # Calculate standard errors
    se_metrics = {
        "accuracy_se": np.std(metrics["accuracy"], ddof=1) / np.sqrt(n_splits),
        "precision_se": np.std(metrics["precision"], ddof=1) / np.sqrt(n_splits),
        "recall_se": np.std(metrics["recall"], ddof=1) / np.sqrt(n_splits),
        "f1_se": np.std(metrics["f1"], ddof=1) / np.sqrt(n_splits)
    }

    return avg_metrics, se_metrics


# Perform cross-validation
cv_metrics_E, se_metrics_E = cross_validate(model_E, env_df_small, compute_metrics, training_args,
                                            random_seed=random_seed)
cv_metrics_S, se_metrics_S = cross_validate(model_S, soc_df_small, compute_metrics, training_args,
                                            random_seed=random_seed)
cv_metrics_G, se_metrics_G = cross_validate(model_G, gov_df_small, compute_metrics, training_args,
                                            random_seed=random_seed)

print("Environmental Model Average Metrics:", cv_metrics_E)
print("Environmental Model SE:", se_metrics_E)
print("Social Model Average Metrics:", cv_metrics_S)
print("Social Model SE:", se_metrics_S)
print("Governance Model Average Metrics:", cv_metrics_G)
print("Governance Model SE:", se_metrics_G)
