import test_accuracy
from densenet import Model
import torch
import database
from PIL import Image
from argparse import ArgumentParser, ArgumentTypeError

def test(num_features):
    model = Model(num_features=num_features)
    db = database.Database("db", model, True)

    data = test_accuracy.TestDataset()

    loader = torch.utils.data.DataLoader(data, batch_size=1,
                                         shuffle=True, num_workers=4, pin_memory=True)

    top_1_acc = 0
    top_5_acc = 0

    for i, image in enumerate(loader):
        names = db.search(Image.open(image[0]).convert('RGB'))

        similar = names[:5]

        already_found_5 = False

        for j in range(len(similar)):
            begin_retr = similar[j].find("/", similar[j].find("/")+1) + 1
            end_retr = similar[j].find("/", begin_retr)

            begin_test = image[0].find("/", image[0].find("/")+1) + 1
            end_test = image[0].find("/", begin_test)

            if similar[j][begin_retr:end_retr] == image[0][begin_test: end_test] \
                and already_found_5 is False:
                top_5_acc += 1
                already_found_5 = True
                if j == 0:
                    top_1_acc += 1

        # print("top 1 accuracy {}, round {}".format((top_1_acc / (i + 1)), i + 1))
        # print("top 5 accuracy {}, round {} ".format((top_5_acc / (i + 1)), i + 1))

    print("top 1 accuracy : ", top_1_acc / data.__len__())
    print("top 5 accuracy : ", top_5_acc / data.__len__())


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
        type=int
    )

    args = parser.parse_args()

    test(args.num_features)
