import model
import data_loader
import torch
from tqdm import tqdm

def train_model(learning_rate, num_epochs, n):
    sim_clr_model = model.SimCLR(500, 100)
    loss_function = model.NXTentLoss(1)
    optimizer = torch.optim.Adam(sim_clr_model.parameters(), lr=learning_rate)
    training_dataset = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')

    running_loss = 0
    for i in tqdm(range(num_epochs)):
        mini_batch = training_dataset.get_mini_batch(n, data_loader.DataAugmentation.augment_gaussian)

        optimizer.zero_grad()

        latent_vectors = sim_clr_model.forward(mini_batch)
        loss = loss_function(latent_vectors)
        loss.backward()
        optimizer.step()

        print(f'Loss: {round(loss.item(), 3)}')

if __name__ == '__main__':
    train_model(1.5, 100, 3)