from database import Database
from argparse import ArgumentParser, ArgumentTypeError
import densenet
from PIL import Image

class ImageRetriever:
    def __init__(self, db_name, model):
        self.db = Database(db_name, model, True)

    def retrieve(self, image):
        return self.db.search(image)

if __name__ == "__main__":
    usage = "python3 add_images.py --path <image_name> [--extractor <algorithm> --db_name <name> --num_features <num>]"

    parser = ArgumentParser(usage)

    parser.add_argument(
        '--path',
        help='path to the image',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used',
        default='densenet'
    )

    parser.add_argument(
        '--db_name',
        help='name of the database',
        default='db'
    )

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
        type=int
    )

    args = parser.parse_args()

    if args.path is None:
        print(usage)
        exit(-1)

    model = None

    if args.extractor == "densenet":
        model = densenet.Model(num_features=args.num_features)

    if model is None:
        print("Unkown feature extractor")
        exit(-1)

    retriever = ImageRetriever(args.db_name, model)

    print(retriever.retrieve(Image.open(args.path).convert('RGB')))
