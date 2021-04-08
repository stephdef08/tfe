import faiss
import csv
import densenet
import torch
import dataset
from PIL import Image
from torchvision import transforms

class Database:
    def __init__(self, filename, model, load=False):
        self.name = filename
        self.embedding_size = 128
        self.nrt_neigh = 10
        self.model = model

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

    def add_dataset(self, data_root):
        data = dataset.AddDataset(data_root)
        loader = torch.utils.data.DataLoader(data, batch_size=32,
                                             num_workers=12, pin_memory=True)

        images_gpu = torch.zeros((32, 224, 224, 3), device='cuda:0')

        with torch.no_grad():
            for i, (images, filenames) in enumerate(loader):
                images_gpu = images.to(device='cuda:0')

                out = self.model(images_gpu)

                self.add(out.numpy(), list(filenames))

        self.save()

    def search(self, x):
        image = transforms.Resize((224, 224))(x)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        out = self.model(image.to(device='cuda:0').view(1, 3, 224, 224))

        _, labels = self.index.search(out.numpy(), self.nrt_neigh)

        with open(self.csv_file, "r") as file:
            lines = file.readlines()

            names = []

            for l in labels[0]:
                names.append(lines[l])

        return  names

    def save(self):
        faiss.write_index(self.index, self.name)

if __name__ == "__main__":
    model = densenet.Model(num_features=128)
    database = Database("db", model)
    database.add_dataset("image_folder/test/38/")

    # image = Image.open("image_folder/test/38/1884_1719624.png").convert('RGB')
    #
    # print(database.search(image))
