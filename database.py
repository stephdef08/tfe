import faiss
import csv
import densenet
import torch
import dataset
from PIL import Image
from torchvision import transforms

class Database:
    def __init__(self, filename, load=False):
        self.name = filename
        self.embedding_size = 32
        self.nrt_neigh = 10

        if load == True:
            self.index = faiss.read_index(filename)
            self.csv_file = filename + ".csv"
        else:
            self.index = faiss.IndexFlatL2(self.embedding_size)
            open(filename + '.csv', 'w')
            self.csv_file = filename + ".csv"

    def add(self, x, names):
        self.index.add(x)

        with open(self.csv_file, "a") as file:
            for n in names:
                file.write(n + "\n")


    def search(self, x):
        _, labels = self.index.search(x, self.nrt_neigh)
        print(labels)

        with open(self.csv_file, "r") as file:
            names = [n for i, n in enumerate(file.readlines()) if i in labels[0]]

        return names

    def save(self):
        faiss.write_index(self.index, self.name)

if __name__ == "__main__":
    model = densenet.Model()
    database = Database("db", True)

    image = Image.open("image_folder/test/38/1884_1719624.png").convert('RGB')
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(image)
    out = model(image.to(device='cuda:0').view(1, 3, 224, 224))
    print(out)

    print(database.search(out.numpy()))

    """
    data_root = "image_folder/test/38/"

    data = dataset.AddDataset(data_root)
    loader = torch.utils.data.DataLoader(data, batch_size=32,
                                         num_workers=12, pin_memory=True)

    images_gpu = torch.zeros((32, 224, 224, 3), device='cuda:0')

    with torch.no_grad():
        for i, (images, filenames) in enumerate(loader):
            images_gpu = images.to(device='cuda:0')

            out = model(images_gpu)

            database.add(out.numpy(), list(filenames))

    database.save()
    """
