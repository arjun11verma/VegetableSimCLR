import model
import data_loader
import torch
import gc
from tqdm import tqdm

USE_CUDA = False

def train_model(learning_rate, num_epochs, n):
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if USE_CUDA else 'cpu')
    sim_clr_model = (model.SimCLR(500, 100)).to(device)
    sim_clr_model.train()

    loss_function = model.NXTentLoss(1, device)
    optimizer = torch.optim.Adam(sim_clr_model.parameters(), lr=learning_rate)
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')

    running_loss = 0
    for i in range(num_epochs):
        mini_batch, batch_labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.pull_from_class)
        mini_batch = mini_batch.to(device)

        optimizer.zero_grad()

        latent_vectors = sim_clr_model(mini_batch)
        loss = loss_function(latent_vectors)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i % 20 == 0):
            print(f'Loss at epoch {i}: {round(running_loss / 20, 3) if i != 0 else running_loss}')
            running_loss = 0
    
    model.save_model(sim_clr_model, f'model_augmentcrop_{num_epochs}_epochs.pt')

def sample_batch(n):
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')
    batch, labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.pull_from_class)

    for i, img in enumerate(batch):
        data_loader.save_image(img, f'Image_{i}_Category_{labels[i][0]}.jpg')

if __name__ == '__main__':
    train_model(0.25, 100, 2)