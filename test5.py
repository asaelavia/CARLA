import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from carla import DataCatalog

np.random.seed(42)
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')
scaler = StandardScaler()
from carla.data.causal_model import CausalModel

scm = CausalModel("sanity-5-lin")
dataset = scm.generate_dataset(10000)
# df_train, df_test = train_test_split(german_credit_df, test_size=0.2, random_state=42)


dataset._df.to_csv('lin5_synth_label')
# data_catalog = GermanCreditCatalog(german_credit_df)


from carla.models.catalog import MLModelCatalog

training_params = {"lr": 0.01, "epochs": 10, "batch_size": 16, "hidden_size": [9, 3]}

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
test_factual = factuals_transformed.iloc[:10]

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
st = time.time()
cfs = CausalRecourse(ml_model, hyperparams,scaler).get_counterfactuals(test_factual)
end = time.time()
print(f'runtime: {end - st}')
cfs.to_csv('cfs_5.csv')
test_factual.to_csv('orig_5.csv')
display(cfs)