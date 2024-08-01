import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from carla import DataCatalog

np.random.seed(42)

continuous = ['income', 'employment_length', 'savings', 'debt']
categorical = ['sex', 'education']
immutables = ['age', 'sex']
target = 'credit_risk'
def generate_german_credit_causal():
    n_samples = 1000

    # Generate exogenous variables
    age = np.random.randint(18, 70, n_samples)  # Age as integer between 18 and 70
    sex = np.random.binomial(1, 0.6, n_samples)
    education = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])

    # Generate endogenous variables based on causal relationships
    income = 1000 + 50 * age + 500 * education + np.random.normal(0, 500, n_samples)
    employment_length = np.maximum(0, age - 18 - 3 * education + np.random.normal(0, 2, n_samples))
    savings = np.maximum(0, 0.1 * income * (age - 18) / 10 + np.random.normal(0, 1000, n_samples))
    debt = np.maximum(0,
                      0.2 * income + 1000 * education - 500 * employment_length + np.random.normal(0, 2000, n_samples))

    # Generate credit score (outcome variable)
    credit_score = (
            0.3 * income / 1000 +
            0.2 * savings / 1000 +
            0.2 * employment_length -
            0.3 * debt / 1000 +
            0.1 * age / 10 +
            0.1 * education +
            np.random.normal(0, 0.5, n_samples)
    )

    # Convert credit score to binary classification
    credit_risk = (credit_score > np.median(credit_score)).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'education': education,
        'income': income,
        'employment_length': employment_length,
        'savings': savings,
        'debt': debt,
        'credit_risk': credit_risk
    })

    # Round and convert to integers
    df['age'] = df['age'].round().astype(int)
    df['employment_length'] = df['employment_length'].round().astype(int)

    # Standardize continuous variables (except age)
    # scaler = StandardScaler()
    # continuous_vars = ['income', 'employment_length', 'savings', 'debt']
    # df[continuous_vars] = scaler.fit_transform(df[continuous_vars])

    return df


# Generate the dataset
# german_credit_df = generate_german_credit_causal()

# print(german_credit_df.head())
# print("\nDataset shape:", german_credit_df.shape)
# print("\nClass distribution:")
# print(german_credit_df['credit_risk'].value_counts(normalize=True))
# print("\nAge statistics:")
# print(german_credit_df['age'].describe())

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')
scaler = StandardScaler()
from carla.data.causal_model import CausalModel

scm = CausalModel("german")
dataset = scm.generate_dataset(10000)
# df_train, df_test = train_test_split(german_credit_df, test_size=0.2, random_state=42)


class GermanCreditCatalog(DataCatalog):
    def __init__(self, df):
        # Split the data into train and test sets
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        super().__init__(
            data_name="german_credit",
            df=df,
            df_train=df_train,
            df_test=df_test,
            scaling_method="MinMax",
            encoding_method="OneHot"
        )

    @property
    def categorical(self):
        return ['sex', 'education']

    @property
    def continuous(self):
        return ['income', 'employment_length', 'savings', 'debt']

    @property
    def immutables(self):
        return ['age', 'sex']

    @property
    def target(self):
        return 'credit_risk'

credit_score = (
            0.3 * dataset.df.x4 / 1000 +
            0.2 * dataset.df.x6 / 1000 +
            0.2 * dataset.df.x5/10 -
            0.3 * dataset.df.x7 / 1000000 +
            0.1 * dataset.df.x1 / 10 +
            0.1 * dataset.df.x3 +
            np.random.normal(0, 0.5, 10000))

# Convert credit score to binary classification
credit_risk = (credit_score > np.median(credit_score)).astype(int)
# Assuming german_credit_df is your generated DataFrame
continuous = ["x1", "x4", "x5", "x6", "x7"]
categorical = ["x2", "x3"]
immutables = ["x2"]
dataset._df[continuous] = scaler.fit_transform(dataset._df[continuous])

dataset._df['label'] = credit_risk
dataset._df_train, dataset._df_test = train_test_split(dataset.df, test_size=0.2, random_state=42)
dataset._df.to_csv('german_synth_label')
# data_catalog = GermanCreditCatalog(german_credit_df)


from carla.models.catalog import MLModelCatalog

training_params = {"lr": 0.01, "epochs": 100, "batch_size": 128, "hidden_size": [9, 3]}

ml_model = MLModelCatalog(
    dataset, model_type="ann", load_online=False, backend="pytorch"
)

ml_model.train(
    learning_rate=training_params["lr"],
    epochs=training_params["epochs"],
    batch_size=training_params["batch_size"],
    hidden_size=training_params["hidden_size"],
    force_train=True
)

from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.causal_recourse import (
    CausalRecourse,
    constraints,
    samplers,
)

# get factuals
factuals_transformed = predict_negative_instances(ml_model, dataset.df)

# Inverse transform the factuals
# factuals_original = data_catalog.inverse_transform(factuals_transformed)
test_factual = factuals_transformed.iloc[:5]

hyperparams = {
    "optimization_approach": "brute_force",
    "num_samples": 10,
    "scm": scm,
    "constraint_handle": constraints.point_constraint,
    "sampler_handle": samplers.sample_true_m0,
}
with open(f'model.pkl', 'wb') as f:
    pickle.dump(ml_model.raw_model, f)
with open(f'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
cfs = CausalRecourse(ml_model, hyperparams,scaler).get_counterfactuals(test_factual)
cfs.to_csv('cfs_5.csv')
test_factual.to_csv('orig_5.csv')
display(cfs)