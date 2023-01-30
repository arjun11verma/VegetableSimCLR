import model
import data_loader
import torch

def train_model(learning_rate):
    sim_clr = model.SimCLR(500, 100)
    nxtent_loss = model.NXTentLoss(20)
    optimizer = torch.optim.Adam(sim_clr.parameters(), lr=learning_rate)
    training_data = data_loader.VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')

    num_epochs = 50
    n = 10
    
    mini_batch = training_data.get_mini_batch(n, data_loader.DataAugmentation.augment_gaussian)
    latent_space = sim_clr.forward(mini_batch)
    loss_function = model.NXTentLoss(temprature=3.5)
    loss = loss_function(latent_space)
    

if __name__ == '__main__':
    train_model(0.05)