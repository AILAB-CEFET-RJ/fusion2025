import torch
from torch.utils.data import Dataset


class Splitter:
    def __init__(self, test_rate, validation_rate):
        self.test_rate = test_rate
        self.validation_rate = validation_rate / (1.0 - self.test_rate)

    def split_test(self, dataset):
        return self.__split(dataset, self.test_rate, first_part=False)

    def split_validation(self, dataset):
        data = self.__split(dataset, self.test_rate, first_part=True)
        return self.__split(data, self.validation_rate, first_part=False)

    def split_train(self, dataset):
        data = self.__split(dataset, self.test_rate, first_part=True)
        return self.__split(data, self.validation_rate, first_part=True)

    def __split(self, dataset, split_rate, first_part):
        if split_rate:
            split = int(dataset.sample.size * (1.0 - split_rate))
            if first_part:
                return dataset[dict(sample=slice(0, split))]
            else:
                return dataset[dict(sample=slice(split, None))]


class NetCDFDataset(Dataset):
    def __init__(
        self,
        dataset,
        test_split=0,
        validation_split=0,
        is_validation=False,
        is_test=False,
        is_2d_model=False,
    ):
        super(NetCDFDataset, self).__init__()
        self.is_2d_model = is_2d_model
        sr = Splitter(test_split, validation_split)
        if is_test:
            data = sr.split_test(dataset)
        elif is_validation:
            data = sr.split_validation(dataset)
        else:
            data = sr.split_train(dataset)
        self.X = torch.from_numpy(data.x.values).float().permute(0, 4, 1, 2, 3)
        self.X = self.X[:, :, :5, :, :]

        CHANNEL = 1
        only_channel_0 = data.y.values[:, :, :, :, :1] if CHANNEL == 1 else data.y.values
        self.y = torch.from_numpy(only_channel_0).float().permute(0, 4, 1, 2, 3)
        if self.is_2d_model:
            self.X = torch.squeeze(self.X)
            self.y = torch.squeeze(self.y)
        del data

    def __getitem__(self, index):
        return (self.X[index, :, :, :, :], self.y[index])

    def __len__(self):
        return self.X.shape[0]
