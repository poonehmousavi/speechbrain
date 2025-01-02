import torch
from torch.utils.data import DataLoader, Sampler
import random
from speechbrain.dataio.dataset import DynamicItemDataset
import speechbrain as sb

class MultiDatasetBatchSampler(Sampler):
    def __init__(self, datasets, probabilities, batch_size):
        """
        A custom sampler for PyTorch DataLoader that ensures each batch
        is drawn from only one dataset, based on the specified probabilities.

        Args:
            datasets (list of Dataset): List of PyTorch Dataset objects.
            probabilities (list of float): Sampling probabilities for each dataset.
            batch_size (int): Number of samples per batch.
        """
        super().__init__(None)
        assert len(datasets) == len(probabilities), "Each dataset must have a corresponding probability."
        assert abs(sum(probabilities) - 1e-0) < 1e-6, "Probabilities must sum to 1."
        
        self.datasets = datasets
        self.probabilities = probabilities
        self.batch_size = batch_size
        self.dataset_lengths = [len(dataset) for dataset in self.datasets]

        # Initialize state for sampling
        self.remaining_indices = {i: list(range(length)) for i, length in enumerate(self.dataset_lengths)}

    def __iter__(self):
        while any(self.remaining_indices.values()):
            # Randomly select a dataset based on probabilities
            dataset_idx = random.choices(
                range(len(self.datasets)), weights=self.probabilities, k=1
            )[0]

            if not self.remaining_indices[dataset_idx]:
                continue  # Skip if this dataset is exhausted

            # Sample a batch of indices from the selected dataset
            batch = []
            for _ in range(self.batch_size):
                if not self.remaining_indices[dataset_idx]:
                    break  # Stop if this dataset is exhausted
                batch.append(self.remaining_indices[dataset_idx].pop())

            if batch:
                yield [(dataset_idx, idx) for idx in batch]

    def __len__(self):
        # Total number of batches across all datasets
        return sum(length // self.batch_size for length in self.dataset_lengths)

    def reset(self):
        """Reset the state for a new epoch."""
        self.remaining_indices = {
            i: list(range(length)) for i, length in enumerate(self.dataset_lengths)
        }



class CombinedDataset(DynamicItemDataset):
    def __init__(self, datasets):
        """
        A combined dataset that allows fetching data from multiple DynamicItemDatasets.

        Args:
            datasets (list of DynamicItemDataset): List of DynamicItemDataset objects.
        """
        self.datasets = datasets
        # self.data = {f"{i}_{key}": value for i, dataset in enumerate(datasets) for key, value in dataset.data.items()}
        # self.data_ids = list(self.data.keys())
        # self.pipeline = None  # Initialize the pipeline dynamically

        # Use the pipeline of the first dataset for initialization
        # if datasets:
        #     self.pipeline = datasets[0].pipeline

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)


    def __getitem__(self, index):
        # Expecting index as (dataset_idx, local_idx)
        dataset_idx, local_idx = index
        print(f"Fetching from dataset {dataset_idx}, index {local_idx}")
        data_id = self.datasets[dataset_idx].data_ids[local_idx]
        data_point = self.datasets[dataset_idx].data[data_id]
        return self.datasets[dataset_idx].pipeline.compute_outputs({"id": data_id, **data_point})


# }
dataset1 = DynamicItemDataset.from_json(
        json_path="data/dataset1.json",
        replacements={"data_root": "data"},)
dataset2 = DynamicItemDataset.from_json(
        json_path="data/dataset2.json",
        replacements={"data_root": "data"},)
dataset3 = DynamicItemDataset.from_json(
        json_path="data/dataset3.json",
        replacements={"data_root": "data"},)
datasets =[dataset1, dataset2, dataset3]
@sb.utils.data_pipeline.takes("label")
@sb.utils.data_pipeline.provides(
        "label"
)
def text_pipeline(label):
        yield label

sb.dataio.dataset.add_dynamic_item(datasets,  text_pipeline)

@sb.utils.data_pipeline.takes("input")
@sb.utils.data_pipeline.provides(
        "input"
)
def audio_pipeline(input):
        yield torch.LongTensor(input).unsqueeze(0)

sb.dataio.dataset.add_dynamic_item(datasets,  audio_pipeline)
sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "label", "input"],
    )


# Define sampling probabilities
probabilities = [0.0, 0.0, 1.0]
batch_size = 2

# Create the combined dataset and batch sampler
datasets = [dataset1, dataset2, dataset3]
combined_dataset = CombinedDataset(datasets)
batch_sampler = MultiDatasetBatchSampler(datasets, probabilities, batch_size)

# Create the DataLoader
dataloader = sb.dataio.dataloader.make_dataloader(combined_dataset, batch_sampler=batch_sampler)

# Iterate through the DataLoader

for i, batch in enumerate(dataloader):
    print(f"Batch {i}:")
    for item in batch:
        print(item)
    if i == 7:  # Limit output for demonstration
        break



