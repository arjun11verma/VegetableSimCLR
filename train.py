import models
import loss_functions
import data_loader
import torch
import gc
from tqdm import tqdm

ACTIVE_CUDA_DEVICE = 1 # -1 if not using cuda

def train_vit_model(learning_rate, num_epochs, n):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device(ACTIVE_CUDA_DEVICE if ACTIVE_CUDA_DEVICE > -1 else 'cpu')

    vit_model = (models.VisualTransformerContrastive(64, 32)).to(device)
    vit_model.train()

    loss_function = loss_functions.NXTentLoss(1, device)
    optimizer = torch.optim.Adam(vit_model.parameters(), lr=learning_rate)
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')

    for i in range(num_epochs):
        mini_batch, batch_labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.pull_from_class)
        mini_batch = mini_batch.to(device)

        optimizer.zero_grad()

        latent_vectors = vit_model(mini_batch)
        loss = loss_function(latent_vectors)
        loss.backward()
        optimizer.step()

        print(f'Loss at epoch {i}: {loss.item()}')
    
    models.save_model(vit_model, f'vit_model_pull_from_class_{num_epochs}_epochs_{n}_classes.pt')

def train_simclr_model(learning_rate, num_epochs, n):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device(ACTIVE_CUDA_DEVICE if ACTIVE_CUDA_DEVICE > -1 else 'cpu')
    sim_clr_model = (models.SimCLR(64, 32)).to(device)
    sim_clr_model.train()

    loss_function = loss_functions.NXTentLoss(1, device)
    optimizer = torch.optim.Adam(sim_clr_model.parameters(), lr=learning_rate)
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')

    for i in range(num_epochs):
        mini_batch, batch_labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.pull_from_class)
        mini_batch = mini_batch.to(device)

        optimizer.zero_grad()

        latent_vectors = sim_clr_model(mini_batch)
        loss = loss_function(latent_vectors)
        loss.backward()
        optimizer.step()

        print(f'Loss at epoch {i}: {loss.item()}')
        
        gc.collect()
        torch.cuda.empty_cache()
    
    models.save_model(sim_clr_model, f'model_pull_from_class_{num_epochs}_epochs_{n}_classes.pt')

def sample_batch(n):
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')
    batch, labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.pull_from_class)

    for i, img in enumerate(batch):
        data_loader.save_image(img, f'Image_{i}_Category_{labels[i][0]}.jpg')

if __name__ == '__main__':
    #train_simclr_model(0.0001, 100, 12)
    train_vit_model(0.0001, 1500, 12)