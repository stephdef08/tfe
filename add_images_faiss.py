from database import Database
from argparse import ArgumentParser, ArgumentTypeError
import densenet

if __name__ == "__main__":
    usage = "python3 add_images.py --path <folder> [--extractor <algorithm> --db_name <name> --num_features <num>]"

    parser = ArgumentParser(usage)

    parser.add_argument(
        '--path',
        help='path to the folder that contains the images to add',
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

    if args.path[-1] != "/":
        print("The path mentionned is not a folder")
        exit(-1)

    model = None

    if args.extractor == "densenet":
        model = densenet.Model(num_features=args.num_features)

    if model is None:
        print("Unkown feature extractor")
        exit(-1)

    database = Database(args.db_name, model, load=True)

    database.add_dataset(args.path)
