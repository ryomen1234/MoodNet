from torchvision import transforms
from torchvision.models import ResNet18_Weights

def get_transform(
        train: bool = True
):
    
    weights = ResNet18_Weights.IMAGENET1K_V1
    
    base_tranform = weights.transforms()
    
    if train:
        return base_tranform
    else:
        return base_tranform
       
     