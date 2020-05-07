from . import fcn8_vgg16, unet2d


def get_base(base_name, exp_dict, n_classes):
    if base_name == "fcn8_vgg16":
        base = fcn8_vgg16.FCN8VGG16(n_classes=n_classes)

    if base_name == "unet2d":
        base = unet2d.UNet(n_channels=3, n_classes=n_classes)

    return base

