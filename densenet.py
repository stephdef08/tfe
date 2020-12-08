import torchvision.models as models

def model():
    return models.densenet161(pretrained=True)
