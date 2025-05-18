import torch.nn as nn
from torchvision import models
from models import HybridResNet # Import custom model

def setup_model(model_name, num_classes):
    """Initializes and configures the specified model."""
    model = None
    if model_name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'inception':
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, num_classes)
        model.AuxLogits.fc = nn.Linear(768, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name.startswith('hybrid_'):
        variant = model_name.split('_', 1)[1]
        if variant == 'full':
            model = HybridResNet(num_classes, use_se=True, use_input_conv=True, use_modified_fc=True, unfreeze_layers=True)
        elif variant == 'no_se':
            model = HybridResNet(num_classes, use_se=False, use_input_conv=True, use_modified_fc=True, unfreeze_layers=True)
        elif variant == 'normal_fc':
            model = HybridResNet(num_classes, use_se=True, use_input_conv=True, use_modified_fc=False, unfreeze_layers=True)
        elif variant == 'no_input_conv':
            model = HybridResNet(num_classes, use_se=True, use_input_conv=False, use_modified_fc=True, unfreeze_layers=True)
        elif variant == 'only_fc':
            model = HybridResNet(num_classes, use_se=True, use_input_conv=True, use_modified_fc=True, unfreeze_layers=False)
        else:
            raise ValueError(f"Unknown hybrid variant: {variant}")
    return model
