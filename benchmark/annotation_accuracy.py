import json

import pandas as pd


def load_ecommerce_data():
    dataset = pd.read_csv("sample_data/ecommerceDataset.csv")
    return dataset


def main():
    file_path = "sample_data/train_wiki.json"
    fewrel_data = load_fewrel_data(file_path)

    # Print the first entry to verify
    print(fewrel_data.keys())
    print(fewrel_data["P931"])
