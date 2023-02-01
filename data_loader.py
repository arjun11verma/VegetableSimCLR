import os
from torchvision.io import read_image, write_jpeg
from torchvision import transforms
import torch
import random

class VegetableDataset():
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_labels = {} # array where each index corresponds to a vegetable label as well as a relative path from self.img_dir

        for vegetable_dir in os.listdir(self.img_dir):
            for image in os.listdir(os.path.join(self.img_dir, vegetable_dir)):
                if vegetable_dir not in self.image_labels: self.image_labels[vegetable_dir] = []
                self.image_labels[vegetable_dir].append(image)
        
        self.image_names = list(self.image_labels.keys())

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, key):
        vegetable_class, idx = key
        img_path = os.path.join(self.img_dir, vegetable_class, self.image_labels[vegetable_class][idx])
        image = read_image(img_path).float()
        return image
    
    def __generate_mini_batch(self, n):
        if (n > len(self.image_names)): ValueError(f'Minibatch unaugmented size {n} is greater than number of classes {len(self.image_names)}')

        class_labels = self.image_names.copy()
        batch_labels = []

        base_class = random.choice(class_labels)
        base_idx = random.randint(0, len(self.image_labels[base_class]) - 1)

        class_labels.remove(base_class)
        batch_labels.append((base_class, base_idx))
        visited = set()
        
        mini_batch = torch.zeros([n * 2, 3, 224, 224], dtype=torch.float)
        mini_batch[0] = self.__getitem__((base_class, base_idx))

        visited.add((base_class, base_idx))
        for i in range(1, n):
            add_class = random.choice(class_labels)
            add_idx = random.randint(0, len(self.image_labels[add_class]) - 1)

            if ((add_class, add_idx) in visited): 
                i -= 1
                continue
            
            class_labels.remove(add_class)
            batch_labels.append((add_class, add_idx))
            visited.add((add_class, add_idx))

            mini_batch[i] = self.__getitem__((add_class, add_idx))
        
        return mini_batch, batch_labels
    
    def __augment_mini_batch(self, mini_batch, batch_labels, augment):
        n = int(mini_batch.shape[0] / 2)
        for i in range(n):
            if (augment == DataAugmentation.pull_from_class):
                augment_class = batch_labels[i][0]
                augment_idx = batch_labels[i][1]
                while (augment_idx == batch_labels[i][1]):
                    augment_idx = random.randint(0, len(self.image_labels[augment_class]) - 1)
                mini_batch[i + n] = self.__getitem__((augment_class, augment_idx))
            else:
                mini_batch[i + n] = augment(mini_batch[i]).float()
        return mini_batch, batch_labels + batch_labels
    
    def get_mini_batch(self, n, augment):
        batch, labels = self.__generate_mini_batch(n) 
        return self.__augment_mini_batch(batch, labels, augment)

class DataAugmentation():
    @staticmethod
    def augment_crop(image):
        return (transforms.RandomResizedCrop((image.shape[1], image.shape[2]))(image))

    @staticmethod
    def augment_gaussian(image):
        return (transforms.GaussianBlur((5, 5), (0.1, 2))(image))

    @staticmethod
    def augment_color(image):
        pass
    
    @staticmethod
    def pull_from_class(image):
        pass

def save_image(image, name):
    write_jpeg(image.to(torch.uint8), '/home/arjun_verma/SimCLR_Implementation/test_outputs/' + name)

if __name__ == '__main__':
    ds = VegetableDataset('/home/arjun_verma/SimCLR_Implementation/data/Vegetable Images/train')

    n = 10
    mini_batch = ds.get_mini_batch(n, DataAugmentation.augment_gaussian)
