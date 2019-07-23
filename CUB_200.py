import os
import pickle
import numpy as np
import PIL.Image
# from tqdm import tqdm
import torch.utils.data


class CUB_200(torch.utils.data.Dataset):
    def __init__(self, file_path, train=True, transform=None, target_transform=None):
        self.file_path = file_path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not (os.path.isfile(os.path.join(self.file_path, 'processed/train.pkl'))
                and os.path.isfile(os.path.join(self.file_path, 'processed/test.pkl'))):
            self.process()

        if self.train:
            print('Read the training dataset...')
            self.train_data, self.train_labels = pickle.load(
                open(os.path.join(self.file_path, 'processed/train.pkl'), 'rb'))
            print('Read successfully!')
        else:
            print('Read the test dataset...')
            self.test_data, self.test_labels = pickle.load(
                open(os.path.join(self.file_path, 'processed/test.pkl'), 'rb'))
            print('Read successfully!')

    def __getitem__(self, index):
        if self.train:
            image, label = self.train_data[index], self.train_labels[index]
        else:
            image, label = self.test_data[index], self.test_labels[index]

        # Transform to PIL.Image format
        image = PIL.Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def process(self):
        image_path = os.path.join(self.file_path, 'raw/CUB_200_2011/images/')
        id_and_path = np.genfromtxt(os.path.join(self.file_path, 'raw/CUB_200_2011/images.txt'), dtype=str)
        id_and_isTrain = np.genfromtxt(os.path.join(self.file_path, 'raw/CUB_200_2011/train_test_split.txt'), dtype=int)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        print('Data preprocessing, storage files')
        # pbar = tqdm(total=len(id_and_path))
        for id in range(len(id_and_path)):
            image = PIL.Image.open(os.path.join(image_path, id_and_path[id, 1]))
            label = int(id_and_path[id, 1][:3]) - 1

            # Converts gray scale to RGB
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')

            np_image = np.array(image)
            image.close()

            if id_and_isTrain[id, 1] == 1:
                train_data.append(np_image)
                train_labels.append(label)
            else:
                test_data.append(np_image)
                test_labels.append(label)
            # pbar.update(1)
        # pbar.close()

        # Store as a.pkl file
        pickle.dump((train_data, train_labels), open(os.path.join(self.file_path, 'processed/train.pkl'), 'wb'))
        pickle.dump((test_data, test_labels), open(os.path.join(self.file_path, 'processed/test.pkl'), 'wb'))
