from torch.utils.data import Dataset

class CustomDatasets(Dataset):
    def __init__(self, X, y, transforms=None):
        super().__init__()

        self.X = X
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        if self.transforms:
            x = self.transforms(x)

        return x, y