
import torchaudio
import speechbrain as sb
import random

def get_function(dataset_name, func_type):
    """
    Dynamically retrieves and executes a function (`dataio_prepare`) for the specified dataset.

    Args:
        dataset_name (str): The name of the dataset.
        func_type (str): The type of function to retrieve (`dataio`).
        **kwargs: Arguments to pass to the function.

    Returns:
        If `func_type` is "dataio", returns the datasets prepared by `{dataset_name}_dataio_prepare`.
    """
    if func_type != "dataio":
        raise ValueError(f"Unsupported function type: {func_type}")

    # Build the expected function name
    func_name = f"{dataset_name}_dataio_prepare"

    # Check if the function exists in the global namespace
    if func_name not in globals():
        raise ValueError(f"Function {func_name} does not exist.")

    # Retrieve the function
    prepare_func = globals()[func_name]

    # Call the function with the provided kwargs
    return prepare_func

def audio_pipeline(wav, sample_rate):
    # Example: resample audio to a target sample rate
    sig = sb.dataio.dataio.read_audio(wav)
    sig = torchaudio.transforms.Resample(orig_freq=sb.dataio.dataio.read_audio_info(wav).sample_rate, new_freq=sample_rate)(sig)
    return sig

def ljspeech_dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    """
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"],
        replacements={"DATA_ROOT": hparams["data_folder"]},
    )
    # Sort training data to speed up training
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        reverse=hparams["sorting"] == "descending",
        key_max_value={"duration": hparams["train_remove_if_longer"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"],
        replacements={"DATA_ROOT": hparams["data_folder"]},
    )
    # Sort validation data to speed up validation
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["valid_remove_if_longer"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"],
        replacements={"DATA_ROOT": hparams["data_folder"]},
    )
    # Sort the test data to speed up testing
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["test_remove_if_longer"]},
    )

    datasets = [train_data, valid_data, test_data]

    # Define audio pipeline
    takes = ["wav"]
    provides = ["sig"]
    def audio_pipeline(wav):
        original_sample_rate = sb.dataio.dataio.read_audio_info(wav).sample_rate
        sig = sb.dataio.dataio.read_audio(wav)
        sig = torchaudio.functional.resample(
            sig, original_sample_rate, hparams["sample_rate"]
        )
        yield sig

    sb.dataio.dataset.add_dynamic_item(
        datasets, audio_pipeline, takes, provides
    )

    # Set output
    sb.dataio.dataset.set_output_keys(datasets, ["id"] + provides)

    return datasets

def combined_dataio_prepare(hparams, datasets_to_include):
    """
    Combines datasets prepared by their respective `dataio_prepare` functions.

    Args:
        hparams (dict): Hyperparameters for dataset preparation.
        datasets_to_include (list): List of dataset names to include.

    Returns:
        dict: Combined train, valid, and test datasets.
    """
    combined_datasets = {"train": [], "valid": [], "test": []}

    for dataset_name in datasets_to_include:
        datasets = get_function(dataset_name, "dataio")

        combined_datasets["train"].append(datasets["train"])
        combined_datasets["valid"].append(datasets["valid"])
        combined_datasets["test"].append(datasets["test"])

    return combined_datasets

def select_dataloader(dataloaders, probabilities):
    """
    Selects a dataloader based on the given probabilities.

    Args:
        dataloaders (list): List of dataloaders (one for each dataset).
        probabilities (list): List of probabilities corresponding to each dataloader.

    Returns:
        A dataloader chosen based on the given probabilities.
    """
    assert len(dataloaders) == len(probabilities), "Dataloaders and probabilities must have the same length."
    assert abs(sum(probabilities) - 1.0) < 1e-6, "Probabilities must sum to 1."

    # Select the index of the dataloader based on probabilities
    selected_index = random.choices(range(len(dataloaders)), weights=probabilities, k=1)[0]

    return dataloaders[selected_index]

def train_with_policies(dataloaders, probabilities, max_iterations, policy="none"):
    """
    Iterates over dataloaders with policies for handling exhausted dataloaders.

    Args:
        dataloaders (list): List of dataloaders (one for each dataset).
        probabilities (list): List of probabilities corresponding to each dataloader.
        max_iterations (int): Maximum number of iterations.
        policy (str): Policy for handling exhausted dataloaders. One of:
            - "none": Remove dataloader from selection after exhaustion.
            - "upsampling": Restart exhausted dataloader and keep it in the selection.
            - "reset": Reset all dataloaders when one is exhausted.
            - "earlystopping": Stop all iterations when one dataloader is exhausted.

    Yields:
        iteration (int): Current iteration index.
        selected_index (int): Index of the selected dataloader.
        batch: Batch from the selected dataloader.
    """
    assert policy in {"none", "upsampling", "reset", "earlystopping"}, "Invalid policy."
    iterators = [iter(dl) for dl in dataloaders]
    active_indices = list(range(len(dataloaders)))  # Active dataloader indices

    for iteration in range(max_iterations):
        if not active_indices:
                # Reset all iterators and re-enable all dataloaders
                iterators = [iter(dl) for dl in dataloaders]
                active_indices = list(range(len(dataloaders)))
                print("Resetting all dataloaders.")

        # Select a dataloader based on probabilities
        selected_index = random.choices(active_indices, weights=[probabilities[i] for i in active_indices], k=1)[0]

        try:
            # Get the next batch from the selected dataloader
            batch = next(iterators[selected_index])
        except StopIteration:
            if policy == "none":
                # Remove the dataloader from the active list
                active_indices.remove(selected_index)
                continue
            elif policy == "upsampling":
                # Restart the iterator and continue
                iterators[selected_index] = iter(dataloaders[selected_index])
                batch = next(iterators[selected_index])
            elif policy == "reset":
                # Reset all iterators
                iterators = [iter(dl) for dl in dataloaders]
                active_indices = list(range(len(dataloaders)))
                batch = next(iterators[selected_index])
            elif policy == "earlystopping":
                # Stop all iterations
                print("Early stopping due to dataloader exhaustion.")
                break

        yield iteration, selected_index, batch

