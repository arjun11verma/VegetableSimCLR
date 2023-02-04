import torch.nn as nn

def compare_vector_similarities(vectors, model, label):
    outputs = model.forward(vectors)
    return nn.CosineSimilarity(dim=0, eps=1e-6)(outputs[0], outputs[1])
