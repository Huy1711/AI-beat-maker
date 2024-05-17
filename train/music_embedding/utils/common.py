import json

def load_dataset(json_metadata_file):
    dataset = []
    with open(json_metadata_file, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset