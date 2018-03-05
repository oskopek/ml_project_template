import argparse
import flags.flags_parser as flags_parser
import importlib


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_file', help='The model file to execute.')
    parser.add_argument('flags_file', help='The flags_file file that is to be executed.')
    # TODO(oskopek): Add parsing of remainder args for overriding flag values.
    parser.add_argument('remainder', nargs=argparse.REMAINDER, help='The remaining command line arguments.')

    args = parser.parse_args()

    flags_parser.parse(args.flags_file, args.remainder)

    model = importlib.import_module(args.model_file)
    model.run()


if __name__ == "__main__":
    main()
