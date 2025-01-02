# Dataset preparation

import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from data.prepare.dataset_registry import get_prepare_function
from speechbrain.utils.distributed import if_main_process

if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
# Prepare datasets
datasets_to_prepare = hparams['datasets']
for dataset_name in datasets_to_prepare:
    print(f"Preparing dataset: {dataset_name}")
    get_prepare_function(dataset_name, **hparams[dataset_name])

# # Combine datasets for training
# from processing.dataio_pipeline import combined_dataio_prepare

# combined_datasets = combined_dataio_prepare(hparams, datasets_to_prepare)

# # Training datasets
# train_data = combined_datasets["train"]
# valid_data = combined_datasets["valid"]
# test_data = combined_datasets["test"]

# # DataLoader
# train_loader = sb.dataio.dataloader.make_dataloader(train_data, batch_size=hparams["batch_size"])
# valid_loader = sb.dataio.dataloader.make_dataloader(valid_data, batch_size=hparams["batch_size"])
# test_loader = sb.dataio.dataloader.make_dataloader(test_data, batch_size=hparams["batch_size"])

# # Training loop
# for batch in train_loader:
#     print(batch)
