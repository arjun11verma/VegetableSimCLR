import torch.nn as nn
import data_loader
import model
import random
import torch
import gc
from tqdm import tqdm

USE_CUDA = True

def compare_vector_similarities(vectors):
    return nn.CosineSimilarity(dim=0, eps=1e-6)(vectors[0], vectors[1])

def eval(simclr_model : model.SimCLR, dataset : data_loader.VegetableDataset, num_negative_pairs, num_positive_pairs):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if USE_CUDA else 'cpu')
    
    simclr_model.eval()
    simclr_model.to(device)

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

        projected_vectors = simclr_model.eval_forward(input)

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

        projected_vectors = simclr_model.eval_forward(input)

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
    simclr_model = model.load_model('model_augmentcrop_100_epochs.pt', 500, 100)
    eval(simclr_model, data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train'), 100, 100)


