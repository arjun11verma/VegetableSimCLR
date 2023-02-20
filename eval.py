import torch.nn as nn
import data_loader
import models
import random
import torch
import gc

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from tqdm import tqdm

ACTIVE_CUDA_DEVICE = 0 # -1 if not using cuda

def compare_vector_similarities(vectors):
    return nn.CosineSimilarity(dim=0, eps=1e-6)(vectors[0], vectors[1])

def tSNEVisualization(model, dataset : data_loader.VegetableDataset, num_vectors):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device(ACTIVE_CUDA_DEVICE if ACTIVE_CUDA_DEVICE > -1 else 'cpu')

    tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=300)
    colors = cm.rainbow(np.linspace(0, 1, len(dataset.image_names)))

    latent_space = {}
    visited_set = set()

    model.eval()
    model.to(device)

    batch_size = 20
    for image_class in tqdm(dataset.image_names):
        latent_space[image_class] = np.zeros((num_vectors, 1000))
        for batch in range(batch_size, num_vectors + 1, batch_size):
            input = torch.zeros((batch_size, 3, 224, 224))

            for i in range(batch_size):
                index = (image_class, random.randint(0, len(dataset.image_labels[image_class]) - 1))
                if (index in visited_set):
                    i -=1
                    continue
                input[i] = dataset[index] # change batch size
            
            input = input.to(device)
            outputs = model.eval_forward(input)
            outputs = outputs.cpu().detach().numpy()
            latent_space[image_class][(batch - batch_size):batch] = outputs
            
        latent_space[image_class] = tsne.fit_transform(latent_space[image_class])
    
    color_idx = 0
    for image_class in latent_space:
        plt.scatter(latent_space[image_class][:, 0], latent_space[image_class][:, 1], color=colors[color_idx])
        color_idx += 1
    
    plt.savefig(f'/home/arjun_verma/SimCLR_Implementation/test_outputs/tSNE_{num_vectors}_vectors_per_class.png')

def eval(model, dataset : data_loader.VegetableDataset, num_negative_pairs, num_positive_pairs):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device(ACTIVE_CUDA_DEVICE if ACTIVE_CUDA_DEVICE > -1 else 'cpu')
    
    model.eval()
    model.to(device)

    average_negative_similarity = 0 # this number should be as close to zero as possible
    average_positive_similarity = 0 # this number should be as close to one as possible

    visited_set = set()
    classes = dataset.image_names
    labels = dataset.image_labels

    for i in tqdm(range(num_negative_pairs)):
        first_class, second_class = random.choice(classes), random.choice(classes)
        first_idx, second_idx = random.randint(0, len(labels[first_class]) - 1), random.randint(0, len(labels[second_class]) - 1)
        first, second = (first_class, first_idx), (second_class, second_idx)

        if (first_class == second_class or (first, second) in visited_set):
            i -= 1
            continue

        visited_set.add((first, second))

        input = torch.zeros((2, 3, 224, 224))
        input[0], input[1] = dataset[first], dataset[second]
        input = input.to(device)

        projected_vectors = model.eval_forward(input)

        average_negative_similarity += compare_vector_similarities(projected_vectors).cpu().item()

    for i in tqdm(range(num_positive_pairs)):
        main_class = random.choice(classes)

        first_idx, second_idx = random.randint(0, len(labels[main_class]) - 1), random.randint(0, len(labels[main_class]) - 1)
        first, second = (main_class, first_idx), (main_class, second_idx)

        if ((first, second) in visited_set):
            i -= 1
            continue

        visited_set.add((first, second))

        input = torch.zeros((2, 3, 224, 224))
        input[0], input[1] = dataset[first], dataset[second]
        input = input.to(device)

        projected_vectors = model.eval_forward(input)

        average_positive_similarity += compare_vector_similarities(projected_vectors).cpu().item()
    
    average_negative_similarity = average_negative_similarity / num_negative_pairs
    average_positive_similarity = average_positive_similarity / num_positive_pairs

    print('------------------------------------------------')
    print(f'{num_negative_pairs} negative sample pairs tested')
    print(f'{num_positive_pairs} positive sample pairs tested')
    print('------------------------------------------------')
    print(f'Average Negative Similarity: {average_negative_similarity}')
    print(f'Average Positive Similarity: {average_positive_similarity}')
    print('------------------------------------------------')

if __name__ == '__main__':
    model = models.load_model('model_augmentcrop_100_epochs.pt', 2048, 2048)
    dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/test')
    eval(model, dataset, 50, 50)
    #tSNEVisualization(model, data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/test'), 10)