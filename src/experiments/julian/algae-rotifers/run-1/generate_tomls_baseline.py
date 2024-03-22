import toml
import os

# Define the path to the output directory
experiment_directory = os.getcwd() + "/src/experiments/julian/algae-rotifers/run-1/runs_baseline"

# Loop through all files in the observed folder
for seed in [10, 17, 42, 93, 97]:
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
                "id": "PytorchLightningLSTMNeuralNetwork",
                "seed": seed,
                "training_params": {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "accelerator": "auto",
                    "show_progress_bar": False,
                    "patience": 5,
                }
            },
            {
                "id": "PytorchLightningGRUNeuralNetwork",
                "seed": seed,
                "training_params": {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "accelerator": "auto",
                    "show_progress_bar": False,
                    "patience": 5,
                }
            },
            {
                "id": "PytorchLightningConvolutionalNeuralNetwork",
                "seed": seed,
                "training_params": {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "accelerator": "auto",
                    "show_progress_bar": False,
                    "patience": 5,
                }
            },
            {
                "id": "PytorchLightningCustomDenseNeuralNetwork",
                "seed": seed,
                "training_params": {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "accelerator": "auto",
                    "show_progress_bar": False,
                    "patience": 5,
                }
            }
        ],
        "data": {
                "synthetic": "/src/data/algae-rotifers/real-world/algae-rotifers-coherent",
                "test_split": 0.79,
                "split_axis": "vertical",
                "time_series": {
                    "input_features": ["algae", "rotifers"],
                    "output_features": ["rotifers"],
                    "input_length": 3,
                    "output_length": 3,
                }
        }, 
    }
    # Write the TOML data to a file with the same name as the observed data file
    toml_filename = "S" + str(seed)
    output_folder = os.getcwd() + experiment_directory + "/" + toml_filename
    print(output_folder)
    output_path = output_folder + "/" + "baseline.toml"
    if not os.path.exists(output_folder):
        # Create a new directory because it does not exist
        os.makedirs(output_folder)
    with open(output_path, "w") as f:
        toml.dump(toml_data, f)
