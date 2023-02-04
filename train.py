import model
import data_loader
import torch
from tqdm import tqdm

# PID: 37589

def train_model(learning_rate, num_epochs, n):
    sim_clr_model = model.SimCLR(500, 100)
    loss_function = model.NXTentLoss(0.25)
    optimizer = torch.optim.Adam(sim_clr_model.parameters(), lr=learning_rate)
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')

    initial_loss, final_loss = 0, 0
    for i in (range(num_epochs)):
        mini_batch, batch_labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.augment_crop)

        optimizer.zero_grad()

        latent_vectors = sim_clr_model.forward(mini_batch)
        loss = loss_function(latent_vectors)
        loss.backward()
        optimizer.step()

        if (i == 0): initial_loss = loss.item()
        if (i == num_epochs - 1): final_loss = loss.item()
    
    model.save_model(sim_clr_model, f'Model_initloss_{round(initial_loss, 3)}_finalloss_{round(final_loss, 3)}.pt')

def sample_batch(n):
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')
    batch, labels = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.augment_crop)

    for i, img in enumerate(batch):
        data_loader.save_image(img, f'Image_{i}_Category_{labels[i][0]}.jpg')

if __name__ == '__main__':
    train_model(0.5, 2500, 8)