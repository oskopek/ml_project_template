import argparse
import flags.flag_parser as flag_parser
import importlib


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('flags_file', nargs=1, help='The flags_file file that is to be executed.')
    parser.add_argument('model_file', nargs=1, help='The model file to execute.')
    parser.add_argument('remainder', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    global FLAGS
    FLAGS = flag_parser.parse(args.flags_file, args.remainder)

    model = importlib.import_module(args.model_file)
    model.run()


if __name__ == "__main__":
    main()
