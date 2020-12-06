import numpy as np #Buffering and Shaping data
import gzip #Accessing Compressed files
import matplotlib.pyplot as plt


class Load():
    def __init__(self, image_file_name, label_file_name, image_size=28, data_amt=100):
        self.image_file_name = image_file_name
        self.label_file_name = label_file_name
        self.image_size = image_size
        self.data_amt = data_amt
        self.data =  [(image, label) \
                        for image, label in zip(self.load_images(), self.load_labels())]


    def load_images(self):
        """ Loads Images

        Gzip: extracts the images pixels through
        the size parameter. This parameter is equal to
        the amount of pixels that make up each images, as
        well as the amount of images

        Numpy: creates a buffer that takes in the
        byte output of the images that have been read through
        gzip. Later, it shapes this buffer into the actual shape
        of the images.

        """
        file = gzip.open(f"data/data/{self.image_file_name}", "r")
        buf = file.read(size=self.image_size * self.image_size * self.data_amt)

        load = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = load.reshape(self.data_amt, self.image_size*self.image_size, 1)

        return data


    def load_labels(self):
        """ Loads Labels

        Gzip: extracts the laels through the size parameter.
        This parameter is equal to the amount of data that
        we want to use.

        Numpy: creates a buffer that takes in the
        byte output of the labels that have been read through
        gzip.
            """
        file = gzip.open(f"data/data/{self.label_file_name}", "r")
        buf = file.read(self.data_amt)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return data

    def __get_data__(self):
        return self.data