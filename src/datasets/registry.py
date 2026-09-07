from src.datasets.pamap2 import Pamap2Dataset
from src.datasets.capture24 import Capture24Dataset
from src.datasets.wisdm import WisdmDataset



DATASET_REGISTRY = {
    "pamap2": Pamap2Dataset,
    "capture24": Capture24Dataset,
    "wisdm" : WisdmDataset,
}
