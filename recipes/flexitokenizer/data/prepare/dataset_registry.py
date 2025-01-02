DATASET_REGISTRY = {
    "librispeech": "librispeech_prepare.prepare_librispeech",
    "iemocap": "iemocap_prepare.prepare_iemocap",
    "ljspeech": "ljspeech_prepare.prepare_ljspeech",
}

def call_prepare_function(dataset_name,**kwargs):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} is not registered.")
    
    module_name, func_name = DATASET_REGISTRY[dataset_name].rsplit(".", 1)
    module = __import__(module_name, fromlist=[func_name])
    # Retrieve the function from the module
    prepare_func = getattr(module, func_name)

    # Call the prepare function with the provided kwargs
    prepare_func(**kwargs)


# params={"data_folder": "path/to/librispeech/data",
# "save_folder": "path/to/librispeech/output",
# "splits": ["train", "valid", "test"],
# "split_ratio": [90, 5, 5],
# "seed": 5678,
# "skip_prep": True}
# t=get_prepare_function("ljspeech",**params)