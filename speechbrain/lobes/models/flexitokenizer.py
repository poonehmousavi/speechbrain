import json
import random
from torch.utils.data import Dataset, DataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataio import load_data_csv, load_data_json

class MultiJsonDataset(DynamicItemDataset):
    """
    A dataset class that manages multiple JSON datasets, samples each with
    specified probabilities, and dynamically creates batches from sampled data.

    Args:
        json_paths (list of str): List of paths to JSON files.
        probabilities (list of float): Sampling probabilities for each JSON dataset.
        dynamic_items (list): Configuration for dynamic items.
        output_keys (list or dict): Keys to include in the output.
    """
    def __init__(self, data, dynamic_items=[], output_keys=[], probabilities=[]):
        # Validate probabilities
        if len(json_paths) != len(probabilities):
            raise ValueError("The number of probabilities must match the number of JSON paths.")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("Probabilities must sum to 1.")
        

        self.probabilities = probabilities

        # Initialize the parent class
        super().__init__(data=data, dynamic_items=dynamic_items, output_keys=output_keys)


    @classmethod
    def from_jsons(
        cls, json_paths, replacements={}, dynamic_items=[], output_keys=[],probabilities=[]
    ):
        """Load a data prep JSON file and create a Dataset based on it."""
        data = [load_data_json(path,replacements) for path in json_paths]
        return cls(data, dynamic_items, output_keys,probabilities)    

    def _sample_dataset(self):
        """Samples a dataset index based on the specified probabilities."""
        return random.choices(self.dataset_ids, weights=self.probabilities, k=1)[0]

    def __getitem__(self, index):
        """Fetches a single data point from the sampled dataset."""
        sampled_dataset_idx = self._sample_dataset()
        dataset = self.datasets[sampled_dataset_idx]

        # Get a random sample from the selected dataset
        data_id = random.choice(list(dataset.keys()))
        data_point = dataset[data_id]

        # Add the ID to the data point
        data_point["id"] = data_id
        return self.pipeline.compute_outputs(data_point)

    def __len__(self):
        """Returns the total number of samples (approximation)."""
        return sum(len(dataset) for dataset in self.datasets)

# Example usage:
# Define the paths to JSON datasets
json_paths = ["data/dataset1.json", "data/dataset2.json", "data/dataset3.json"]
probabilities = [0.5, 0.3, 0.2]  # Probabilities for each dataset

# Dynamic items and output keys
dynamic_items = [
    {"func": lambda text: text.split(), "takes": ["input"], "provides": "tokens"},
    {"func": lambda tokens: len(tokens), "takes": ["tokens"], "provides": "length"},
]
output_keys = ["id", "tokens", "length"]

# Initialize the dataset
train_data = DynamicItemDataset.from_json(
        json_path="data/dataset1.json",
        replacements={"data_root": "data"},
    )
dataset = MultiJsonDataset.from_jsons( json_paths=json_paths,
        replacements={"data_root": "data"},
        probabilities=probabilities)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, collate_fn=lambda x: x)

# Fetch a batch
batch = next(iter(dataloader))
print(batch)
