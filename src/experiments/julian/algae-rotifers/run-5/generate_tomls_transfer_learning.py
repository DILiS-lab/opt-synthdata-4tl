import toml
import os

# Define the path to the folder containing the observed data
synthetic_folder = "src/data/algae-rotifers/synthetic"

# Define the path to the output directory
experiment_directory = (
    "/src/experiments/julian/algae-rotifers/run-5/runs_transfer_learning"
)

# Loop through all files in the observed folder
for filename in os.listdir(synthetic_folder):
    if filename.startswith("TS1"):
        seed = int(filename.split("_S")[1])
        # Create a new TOML file with the modified observed path
        toml_data = {
            "plugins": [
                "src.models.freezing_cnn",
                "src.models.freezing_dnn",
                "src.models.freezing_lstm",
                "src.models.freezing_gru",
                "src.metrics.naive_forecasting_error",
            ],
            "metrics": [
                "mean_absolute_scaled_error",
                "prediction_trend_accuracy",
                "mean_directional_accuracy",
                "mean_absolute_error",
                "mean_absolute_scaled_error",
                "mean_squared_error",
                "root_mean_squared_error",
                "normalized_root_mean_squared_error",
                "mean_absolute_percentage_error",
                "naive_forecasting_error",
            ],
            "models": [
                {
                    "id": "PytorchLightningTransferLearningConvolutionalNeuralNetwork",
                    "seed": seed,
                    "training_params": {
                        "epochs": 30,
                        "finetuning_epochs": 5,
                        "learning_rate": 0.001,
                        "finetuning_learning_rate": 0.0001,
                        "batch_size": 64,
                        "accelerator": "gpu",
                        "show_progress_bar": True,
                        "patience": 5,
                    },
                },
                {
                    "id": "PytorchLightningTransferLearningCustomDenseNeuralNetwork",
                    "seed": seed,
                    "training_params": {
                        "epochs": 30,
                        "finetuning_epochs": 5,
                        "learning_rate": 0.001,
                        "finetuning_learning_rate": 0.0001,
                        "batch_size": 64,
                        "accelerator": "gpu",
                        "show_progress_bar": True,
                        "patience": 5,
                    },
                },
                {
                    "id": "PytorchLightningTransferLearningLSTMNeuralNetwork",
                    "seed": seed,
                    "training_params": {
                        "epochs": 30,
                        "finetuning_epochs": 5,
                        "learning_rate": 0.001,
                        "finetuning_learning_rate": 0.0001,
                        "batch_size": 64,
                        "accelerator": "gpu",
                        "show_progress_bar": True,
                        "patience": 5,
                    },
                },
                {
                    "id": "PytorchLightningTransferLearningGRUNeuralNetwork",
                    "seed": seed,
                    "training_params": {
                        "epochs": 30,
                        "finetuning_epochs": 5,
                        "learning_rate": 0.001,
                        "finetuning_learning_rate": 0.0001,
                        "batch_size": 64,
                        "accelerator": "gpu",
                        "show_progress_bar": True,
                        "patience": 5,
                    },
                },
            ],
            "data": {
                "synthetic": os.path.join("/", synthetic_folder, filename),
                "observed": "/src/data/algae-rotifers/real-world/algae-rotifers-incoherent",
                "test_split": 0.76,
                "split_axis": "vertical",
                "export_path": os.path.join(
                    f"./results/algae-rotifers-incoherent/TL/{filename}"
                ),
                "time_series": {
                    "input_features": ["algae", "rotifers"],
                    "output_features": ["rotifers"],
                    "input_length": 3,
                    "output_length": 3,
                },
            },
        }
        # Write the TOML data to a file with the same name as the observed data file
        toml_filename = os.path.splitext(filename)[0]
        output_folder = os.getcwd() + experiment_directory + "/" + toml_filename
        output_path = output_folder + "/" + "transfer.toml"
        if not os.path.exists(output_folder):
            # Create a new directory because it does not exist
            os.makedirs(output_folder)
        with open(output_path, "w") as f:
            toml.dump(toml_data, f)
