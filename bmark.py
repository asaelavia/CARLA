import pickle

from tensorboard.notebook import display

from carla import Benchmark
import carla.evaluation.catalog as evaluation_catalog
from carla.data.catalog import OnlineCatalog
from carla.data.catalog.online_catalog import AdultCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
import carla.recourse_methods.catalog as recourse_catalog

import warnings
warnings.filterwarnings("ignore")

data_name = "adult"
# dataset = OnlineCatalog(data_name)
dataset = AdultCatalog(data_name,encoding_method='Identity')

# load catalog model
model_type = "ann"
training_params = {"lr": 0.002, "epochs": 20, "batch_size": 128, "hidden_size": [18, 9, 3]}

ml_model = MLModelCatalog(
    dataset,
    model_type="ann",
    load_online=False,
    backend="pytorch"
)

ml_model.train(
    learning_rate=training_params["lr"],
    epochs=training_params["epochs"],
    batch_size=training_params["batch_size"],
    hidden_size=training_params["hidden_size"]
)

hyperparams = {
    "data_name": data_name,
    "vae_params": {
        "layers": [sum(ml_model.get_mutable_mask()), 16, 8],
    },
}
# ml_model = MLModelCatalog(
#     dataset,
#     model_type="forest",
#     load_online=False,
#     backend="sklearn"
# )
#
# ml_model.train(
#     learning_rate=training_params["lr"],
#     epochs=training_params["epochs"],
#     batch_size=training_params["batch_size"],
#     hidden_size=training_params["hidden_size"]
# )


# define your recourse method

recourse_method = recourse_catalog.Face(ml_model,hyperparams)
recourse_method.mode = "epsilon"

# coeffs, intercepts = None, None
# recourse_method = recourse_catalog.ActionableRecourse(
#         ml_model, coeffs=coeffs, intercepts=intercepts)

# hyperparams = {
#     "data_name": data_name,
#     "vae_params": {
#         "layers": [sum(ml_model.get_mutable_mask()), 16, 8],
#     },
# }

# recourse_method = recourse_catalog.CRUD(ml_model, hyperparams)

# get some negative instances
factuals = predict_negative_instances(ml_model, dataset.df)
factuals = factuals[:10]

# find counterfactuals
counterfactuals = recourse_method.get_counterfactuals(factuals)

# first initialize the benchmarking class by passing
# black-box-model, recourse method, and factuals into it
# benchmark = Benchmark(ml_model, recourse_method, factuals)

# now you can decide if you want to run all measurements
# or just specific ones.
# evaluation_measures = [
#     evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
#     evaluation_catalog.Distance(benchmark.mlmodel),
#     evaluation_catalog.SuccessRate(),
#     evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
#     evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
#     evaluation_catalog.AvgTime({"time": benchmark.timer}),
# ]

# with open('crud.pkl','rb') as f:
#     res = pickle.load(f)

# Usage
original_factuals = dataset.inverse_transform(factuals)
original_counterfactuals = dataset.inverse_transform(counterfactuals.dropna())

# now run all implemented measurements and create a
# DataFrame which consists of all results
# results = benchmark.run_benchmark(evaluation_measures)
pass