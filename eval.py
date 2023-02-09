import torch.nn as nn
import data_loader
import model
import random
import torch

def compare_vector_similarities(vectors):
    return nn.CosineSimilarity(dim=0, eps=1e-6)(vectors[0], vectors[1])

def eval(simclr_model : model.SimCLR, dataset : data_loader.VegetableDataset, num_negative_pairs, num_positive_pairs):
    simclr_model.eval()

    average_negative_similarity = 0 # this number should be as close to zero as possible
    average_positive_similarity = 0 # this number should be as close to one as possible

    visited_set = set()
    classes = dataset.image_names
    labels = dataset.image_labels

    for i in range(num_negative_pairs):
        first_class, second_class = random.choice(classes), random.choice(classes)
        first_idx, second_idx = random.randint(0, len(labels[first_class]) - 1), random.randint(0, len(labels[second_class]) - 1)
        first, second = (first_class, first_idx), (second_class, second_idx)

        if (first_class == second_class or (first, second) in visited_set):
            i -= 1
            continue

        visited_set.add((first, second))

        input = torch.zeros((2, 3, 224, 224))
        input[0], input[1] = dataset[first], dataset[second]
        average_negative_similarity += compare_vector_similarities(simclr_model.forward(input))

    for i in range(num_positive_pairs):
        main_class = random.choice(classes)

        first_idx, second_idx = random.randint(0, len(labels[main_class]) - 1), random.randint(0, len(labels[main_class]) - 1)
        first, second = (main_class, first_idx), (main_class, second_idx)

        if ((first, second) in visited_set):
            i -= 1
            continue

        visited_set.add((first, second))

        input = torch.zeros((2, 3, 224, 224))
        input[0], input[1] = dataset[first], dataset[second]
        average_positive_similarity += compare_vector_similarities(simclr_model.forward(input))
    
    average_negative_similarity = average_negative_similarity.item() / num_negative_pairs
    average_positive_similarity = average_positive_similarity.item() / num_positive_pairs

    with open('/home/arjun_verma/SimCLR_Implementation/models/evaloutput.txt', 'a') as f:
        f.write('------------------------------------------------\n')
        f.write(f'{num_negative_pairs} negative sample pairs tested\n')
        f.write(f'{num_positive_pairs} positive sample pairs tested\n')
        f.write('------------------------------------------------\n')
        f.write(f'Average Negative Similarity: {average_negative_similarity}\n')
        f.write(f'Average Positive Similarity: {average_positive_similarity}\n')
        f.write('------------------------------------------------\n')

if __name__ == '__main__':
    simclr_model = model.load_model('model_pullfromclass_2000_epochs.pt', 500, 100)
    eval(simclr_model, data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train'), 500, 500)


