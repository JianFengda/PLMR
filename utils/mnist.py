import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .randaugment import RandomAugment
from .utils_algo import generate_uniform_cv_candidate_labels
import os
import os.path

def load_mnist(partial_rate, batch_size):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    train_data, train_labels = torch.load(os.path.join('./data/mnist/',processed_folder,training_file))
    train_data = train_data.numpy()

    test_data, test_labels = torch.load(os.path.join('./data/mnist/', processed_folder,test_file))

    test_dataset = Test_Augmentation(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size * 4, shuffle=False,
                                              num_workers=4)

    partialY = generate_uniform_cv_candidate_labels(train_labels, partial_rate)
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), train_labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = MNIST_Augmentation(train_data, partialY.float(), train_labels.float())

    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset,
                                                              batch_size=batch_size,
                                                              shuffle=True,
                                                              num_workers=4,
                                                              pin_memory=True,
                                                              drop_last=True)
    return partial_matrix_train_loader, partialY, test_loader
class Test_Augmentation(Dataset):
    def __init__(self, images, true_labels):
        self.images = images
        self.true_labels = true_labels
        self.transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    def __len__(self):
        return len(self.true_labels)
    def __getitem__(self, index):
        img,true = self.images[index], self.true_labels[index]
        img = img.numpy()
        if self.transform is not None:
            img = self.transform(img)
        return img,true

class MNIST_Augmentation(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
        self.strong_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                RandomAugment(3, 5),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]


        return each_image_w, each_image_s, each_label, each_true_label, index, each_image