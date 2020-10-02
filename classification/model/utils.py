
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2


__all__ = ["get_model"]

models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2
}


def get_model(name: str, num_classes: int, **kwargs):
    print(kwargs)
    if name.lower() not in models:
        raise ValueError("no model named {}, should be one of {}".format(name, ' '.join(models)))

    return models.get(name.lower())(num_classes, **kwargs)
