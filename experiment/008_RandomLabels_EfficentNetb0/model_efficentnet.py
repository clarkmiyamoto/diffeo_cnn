import torchvision.models as models

def init_model():
    return models.efficientnet_b0(pretrained=False)