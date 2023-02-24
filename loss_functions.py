import torch
import torch.nn as nn

class NXTentLoss(nn.Module):
    def __init__(self, temprature, device):
        super().__init__()
        self.temprature = temprature
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.device = device
        self.logSoftmax = nn.LogSoftmax(0)

    def __generate_similarity_matrix(self, projected_vectors):
        """
        Generates the similarity matrix. My implementation has the first n vectors as the original n images,
        and the next n vectors as the augmented images. Thus, the value 1.0 would be found at (n, n) and the
        similarity value for the unaugmented/augmented image pair would be found at (i, n + i) or (i, i - n)
        for any index i depending on whether i is less than or greater than n respectively
        """
        similarity_matrix = torch.zeros([projected_vectors.shape[0], projected_vectors.shape[0] - 1], dtype=torch.float64).to(self.device)

        for i, x in enumerate(projected_vectors):
            if (i == projected_vectors.shape[0] - 1): break
            
            x = x[None, :]
            similarity = self.cosine_similarity(x, projected_vectors[(i + 1):])
            similarity_matrix[i, i:] = similarity
            similarity_matrix[(i + 1):, i] = similarity

        return similarity_matrix
    
    def __get_loss_pair(self, similarity_row, positive_index):
        n = int(similarity_row.shape[0] / 2) + 1
        increment = positive_index + (n - 1) if (positive_index < n) else (positive_index - n)

        log_softmax_output = self.logSoftmax(similarity_row / self.temprature)

        one_hot_selector = torch.zeros(similarity_row.shape[0]).to(self.device).double()
        one_hot_selector[increment] = 1
        
        loss_value = -1 * torch.dot(log_softmax_output, one_hot_selector)

        return loss_value
    
    def forward(self, projected_vectors):
        similarity_matrix = self.__generate_similarity_matrix(projected_vectors)
        loss = 0
        for i in range(similarity_matrix.shape[0]):
            loss += self.__get_loss_pair(similarity_matrix[i], i)
        return loss