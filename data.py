import time

from data_preprocessing.graph_construction_mimic import construct_graph

dataset = "mimic"
dataset_path = "/sda/encounter_data"
construct_graph(9, dataset, dataset_path)
