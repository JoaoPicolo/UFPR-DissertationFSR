import sys
import argparse

from keras import Model

from face_recognition import get_face_recognition

sys.path.append("..")
from shared.utils import get_parser
from shared.plots import get_dataset_split, manipulate_dataset

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path to the images directory")
    args = parser.parse_args()

    return args

def custom_loss_function(y_true, y_pred):
    pass

# Trains the face recognition model
def main():
    args = get_parser()

    # Loads the dataset and splits
    lr_shape = (54, 44, 3)
    hr_shape = (216, 176, 6)
    train, validation, test = get_dataset_split(args.path, (hr_shape[0], hr_shape[1]), 0.6, 0.2, 0.2)
    train_hr, validation_hr, test_hr = manipulate_dataset(train, validation, test)
    train_lr, validation_lr, test_lr = manipulate_dataset(train, validation, test, resize=True, resize_shape=(lr_shape[0], lr_shape[1]))

    # Get models
    model = get_face_recognition(input_shape=hr_shape)



if __name__ == "__main__":
    main()