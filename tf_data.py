import tensorflow as tf
import numpy as np


def main():
    print(np.random.permutation([1, 2, 3]))
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


if __name__ == '__main__':
    main()