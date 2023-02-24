import torch
import torch.nn as nn
import torchvision
import torchvision.models.vision_transformer as ViT

class VisualTransformerContrastive(nn.Module):
    def __init__(self, proj_hidden_features, proj_output_features):
        super(VisualTransformerContrastive, self).__init__()
        self.encoder = ViT.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        self.linear_projection = nn.Sequential(
            nn.Linear(1000, proj_hidden_features),                                   
            nn.ReLU(),
            nn.Linear(proj_hidden_features, proj_output_features)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.linear_projection(x)
    
    def eval_forward(self, x):
        return self.encoder(x)

class BaseResNet(nn.Module):
    def __init__(self):
        super(BaseResNet, self).__init__()
        self.conv_layers = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)

    def forward(self, x):
        return self.conv_layers(x)

    def eval_forward(self, x):
        return self.conv_layers(x)

class SimCLR(nn.Module):
    def __init__(self, proj_hidden_features, proj_output_features):
        super(SimCLR, self).__init__()
        self.conv_layers = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        for param in self.conv_layers.parameters():
            param.requires_grad = True
            
        self.linear_projection = nn.Sequential(
            nn.Linear(1000, proj_hidden_features),                                   
            nn.ReLU(),
            nn.Linear(proj_hidden_features, proj_output_features)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.linear_projection(x)

    def eval_forward(self, x):
        return self.conv_layers(x)

class TransferClassifier(nn.Module):
    def __init__(self, proj_hidden_features, proj_output_features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1000, proj_hidden_features),
                                               nn.ReLU(),
                                               nn.Linear(proj_hidden_features, proj_output_features),
                                               nn.LogSoftmax())

    def forward(self, x):
        return self.layers(x)

def save_model(model, name):
    torch.save(model.state_dict(), '/home/arjun_verma/SimCLR_Implementation/models/' + name)

def load_model(name, return_model):
    return_model.load_state_dict(torch.load('/home/arjun_verma/SimCLR_Implementation/models/' + name))
    return return_model

