from random import randrange
from torch.utils.data import Dataset
import numpy as np


def random_corners(new_size, img):
    # randomly select a top left corner to use for img
    x_min, y_min = randrange(new_size - img.shape[0]), randrange(new_size - img.shape[0])
    # compute bottom right corner
    x_max, y_max = x_min + img.shape[0], y_min + img.shape[0]
    return x_min, y_min, x_max, y_max  # return top left, bottom right coordinates


class CustomMnistDataset_OL(Dataset):
    def __init__(self, df, test=False):
        self.df = df
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # img into 28x28 array
        image = np.reshape(np.array(self.df[idx][0]), (28, 28))


        new_size = 90
        # create the new image
        new_img = np.zeros((new_size, new_size))  # images will be 90x90
        x_min, y_min, x_max, y_max = random_corners(new_size, image)

        new_img[x_min:x_max, y_min:y_max] = image

        new_img = np.reshape(new_img, (1, 90, 90))  # batch

        # label = [int(self.df[idx][-1]), np.array([x_min, y_min, x_max, y_max]).astype('float32')]
        label = int(self.df[idx][-1])
        sample = {"image": new_img, "label": label}

        return sample['image'], sample['label']