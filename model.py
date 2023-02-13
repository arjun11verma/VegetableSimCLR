import torch
import torch.nn as nn
import torchvision

# env = averma_dev_env

class SimCLR(nn.Module):
    def __init__(self, proj_hidden_features, proj_output_features):
        super(SimCLR, self).__init__()
        self.conv_layers = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        for param in self.conv_layers.parameters():
            param.requires_grad = True
        self.linear_projection = nn.Sequential(nn.Linear(1000, proj_hidden_features),
                                               nn.ReLU(),
                                               nn.Linear(proj_hidden_features, proj_output_features))

    def forward(self, x):
        x = self.conv_layers(x)
        return self.linear_projection(x)

    def eval_forward(self, x):
        return self.conv_layers(x)

class NXTentLoss(nn.Module):
    def __init__(self, temprature, device):
        super().__init__()
        self.temprature = temprature
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.device = device

    def __generate_similarity_matrix(self, projected_vectors):
        """
        Generates the similarity matrix. My implementation has the first n vectors as the original n images,
        and the next n vectors as the augmented images. Thus, the value 1.0 would be found at (n, n) and the
        similarity value for the unaugmented/augmented image pair would be found at (i, n + i) or (i, i - n)
        for any index i depending on whether i is less than or greater than n respectively
        """
        similarity_matrix = torch.zeros([projected_vectors.shape[0], projected_vectors.shape[0]], dtype=torch.float64).to(self.device)
        for i, x in enumerate(projected_vectors):
            x = x[None, :]
            similarity = self.cosine_similarity(x, projected_vectors[i:])
            similarity_matrix[i, i:] = similarity
            similarity_matrix[i:, i] = similarity
        return similarity_matrix
    
    def __get_loss_pair(self, similarity_row, positive_index):
        n_increment = int(similarity_row.shape[0] / 2)
        if (positive_index >= n_increment): n_increment *= -1

        similarity_row = torch.exp(similarity_row / self.temprature)
        sum_selection = torch.ones(similarity_row.shape[0]).to(self.device)
        
        positive_similarity = similarity_row[positive_index + n_increment]
        sum_selection[positive_index] = 0
        similarity_row = similarity_row * sum_selection

        return -1 * torch.log(positive_similarity / torch.sqrt(torch.sum(similarity_row)))
    
    def forward(self, projected_vectors):
        similarity_matrix = self.__generate_similarity_matrix(projected_vectors)
        loss = 0
        for i in range(similarity_matrix.shape[0]):
            loss += self.__get_loss_pair(similarity_matrix[i], i)
        return loss

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

def load_model(name, proj_hidden_features, proj_output_features):
    return_model = SimCLR(proj_hidden_features, proj_output_features)
    return_model.load_state_dict(torch.load('/home/arjun_verma/SimCLR_Implementation/models/' + name))
    return return_model

