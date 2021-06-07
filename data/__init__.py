from data.base_dataset import BaseDataset
from torchvision import transforms as tt
from data.flow_dataset import PlantDataset, IperDataset,Human36mDataset, VegetationDataset, LargeVegetationDataset, BairDataset, UIDummyDataset, TaichiDataset
# from data.cvp_dataset import IperCVPDataset, TaichiCVPDataset
# from data.ci_dataset import CIPlantDataset, CIIperDataset, CITaichiDataset, CIH36mDataset
# from models.baselines.cvp.data_utils import imagenet_preprocess


# add key value pair for datasets here, all datasets should inherit from base_dataset
__datasets__ = {"IperDataset": IperDataset,
                "PlantDataset": PlantDataset,
                "Human36mDataset": Human36mDataset,
                "VegetationDataset": VegetationDataset,
                "LargeVegetationDataset": LargeVegetationDataset,
                "TaichiDataset": TaichiDataset,
                # "IperCVPDataset": IperCVPDataset,
                # "TaichiCVPDataset": TaichiCVPDataset,
                # "CIPlants": CIPlantDataset,
                # "CIper": CIIperDataset,
                # "CITaichi": CITaichiDataset,
                # "CIH36m": CIH36mDataset
                }

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# returns only the class, not yet an instance
def get_transforms(config):
    return {
        "PlantDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "IperDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "Human36mDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "VegetationDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "LargeVegetationDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "BairDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "DummyDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "TaichiDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),

    }


def get_dataset(config, custom_transforms=None):
    dataset = __datasets__[config["dataset"]]
    if custom_transforms is not None:
        print("Returning dataset with custom transform")
        transforms = custom_transforms
    else:
        transforms = get_transforms(config)[config["dataset"]]
    return dataset, transforms

