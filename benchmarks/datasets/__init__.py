from .synbols import Synbols
from torchvision import transforms as tt

def get_dataset(split, dataset_spec):
    if dataset_spec["name"] == "synbols":
        transform = tt.Compose([tt.ToPILImage(), tt.ToTensor()])
        return Synbols(dataset_spec["path"], split, dataset_spec["task"], transform)
