### Format Data
inference_transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
my_inference_transforms = v2.Compose([
    lambda x: x.convert('RGB'),
    inference_transform,
])


### Import Data
data_root = '/vast/work/public/ml-datasets/imagenet/imagenet-test.sqf'

# Actual dataset with TorchTensors
dataset = torchvision.datasets.ImageNet(data_root, split = 'val', transform = my_inference_transforms)

# Copy of dataset for visualization
dataset_visualization = torchvision.datasets.ImageNet(data_root, split = 'val', transform = v2.CenterCrop(224))