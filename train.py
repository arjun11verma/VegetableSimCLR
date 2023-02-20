import models
import loss_functions
import data_loader
import torch
import gc
from tqdm import tqdm

ACTIVE_CUDA_DEVICE = 0 # -1 if not using cuda

def train_model(learning_rate, num_epochs, n):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device(ACTIVE_CUDA_DEVICE if ACTIVE_CUDA_DEVICE > -1 else 'cpu')
    sim_clr_model = (models.SimCLR(2048, 2048)).to(device)
    sim_clr_model.train()

    loss_function = loss_functions.NXTentLoss(1, device)
    optimizer = torch.optim.Adam(sim_clr_model.parameters(), lr=learning_rate)
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')

    running_loss = 0
    for i in range(num_epochs):
        mini_batch, batch_labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.augment_crop)
        mini_batch = mini_batch.to(device)

        optimizer.zero_grad()

        latent_vectors = sim_clr_model(mini_batch)
        loss = loss_function(latent_vectors)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i % 1 == 0):
            print(f'Loss at epoch {i}: {round(running_loss / 1, 3) if i != 0 else running_loss}')
            running_loss = 0
        
        gc.collect()
        torch.cuda.empty_cache()
    
    models.save_model(sim_clr_model, f'model_augmentcrop_{num_epochs}_epochs.pt')

def sample_batch(n):
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')
    batch, labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.pull_from_class)

    for i, img in enumerate(batch):
        data_loader.save_image(img, f'Image_{i}_Category_{labels[i][0]}.jpg')

if __name__ == '__main__':
    train_model(0.0001, 100, 6)