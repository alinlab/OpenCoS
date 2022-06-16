from .resnet import *
from .cifar_resnet import *
from .resnet_auxbn import *
from .cifar_resnet_auxbn import *
from .wide_resnet import *
from .wide_resnet_auxbn import *
from .meta_resnet import *

def load_model(name, num_classes=10, pretrained=False, divide=False, **kwargs):
    model_dict = globals()
    if 'wide' in name or 'Wide' in name:
        if divide:
            model = model_dict[name](28, 2, num_classes=num_classes, divide=divide, **kwargs)
        else:
            model = model_dict[name](28, 2, num_classes=num_classes, **kwargs)
    else:
        if divide:
            model = model_dict[name](pretrained=pretrained, num_classes=num_classes, divide=divide, **kwargs)
        else:
            model = model_dict[name](pretrained=pretrained, num_classes=num_classes, **kwargs)
    return model
