import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path to the images directory")
    args = parser.parse_args()

    return args