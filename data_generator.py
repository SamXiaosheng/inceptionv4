import gzip

import numpy as np

class MLData:
    '''
    Base class for a container for machine learning data
    '''

    def __init__(self, inputs, outputs):
        self.read_data(inputs, outputs)

        return


    def read_data(self, inputs, outputs):
        '''
        Reads data into the object's internal memory
        '''

        if inputs is None or outputs is None:
            pass

        else:
            self.inputs = inputs
            self.outputs = outputs

            self.num_samples = inputs.shape[0]
            self.num_rows = inputs.shape[1]
            self.num_cols = inputs.shape[2]
            self.num_channels = inputs.shape[3]
            self.num_classes = outputs.shape[1]

            self.unused_ix = np.array(range(self.num_samples))

        return


    def get_batch(self, num):
        '''
        Retrieves a batch of inputs and outputs for training. Note that
        samples are not ordered and are drawn randomly from the complete
        dataset

        Keyword arguments:
        num -- number of samples in the batch
        '''

        if self.unused_ix.size < num:
            ix = np.copy(self.unused_ix)

            batch_inputs1 = self.inputs[ix, :, :, :]
            batch_outputs1 = self.inputs[ix, :]

            self.unused_ix = np.setdiff1d(np.array(range(self.num_samples)), ix)

            num = num - ix.size

            batch_inputs1, batch_outputs2 = get_batch(num)
            batch_inputs = np.concatenate(batch_inputs1, batch_inputs2, axis=0)
            batch_outputs = np.concatenate(batch_outputs1, batch_outputs2, axis=0)

            self.unused_ix = np.concatenate(self.unused_ix, ix)

            return batch_inputs, batch_outputs

        else:
            ix = np.random.choice(self.unused_ix, num, replace=False)
            batch_inputs = self.inputs[ix, :, :, :]
            batch_outputs = self.outputs[ix, :]

            self.unused_ix = np.setdiff1d(self.unused_ix, ix)

            return batch_inputs, batch_outputs

    
    def split(self, num):
        '''
        Splits the dataset into two components, with the first being of size
        num

        Keyword arguments:
        num -- the number of samples in the first outputted MLData instance
        '''

        ix1 = np.random.choice(range(self.num_samples), num, replace=False)
        ix2 = np.setdiff1d(range(self.num_samples), ix1)

        output1 = MLData(self.inputs[ix1, :, :, :], self.outputs[ix1, :])
        output2 = MLData(self.inputs[ix2, :, :, :], self.outputs[ix2, :])

        return output1, output2



class MNISTData(MLData):
    '''
    A container class for MNIST data
    '''

    def __init__(self, images_path, labels_path):
        '''
        Initializes the class and decompresses and reads the idx file into
        an np.ndarray. Assumes that the file format is as directly taken from
        http:/yann.lecun.com/exdb/mnist/

        Keyword arguments:
        images_path -- the path to the images file
        labels_path -- the path to the labels file

        The image array is indexed as [sample, row, col, ch]. The labels array
        is indexed as [sample, class]
        '''

        f = gzip.open(images_path)
        images_b = f.read()
        
        f = gzip.open(labels_path)
        labels_b = f.read()

        '''
        =================
        MNIST FILE FORMAT
        =================
        TRAINING IMAGES
        0000-0003 : magic number
        0004-0007 : number of images
        0008-0011 : number of rows
        0012-0015 : number of columns
        0016      : pixel
        0017      : pixel
        ...
        xxxx      : pixel
        
        TRAINING LABELS
        0000-0003 : magic number
        0004-0007 : number of items
        0008      : label
        0009      : label
        ...
        xxxx      : label
        '''
        
        # The bytes here need to be reversed because they are stored big endian
        # while numpy processes numbers little endian
        num_row = int(np.frombuffer(images_b[8:12][::-1], dtype=np.int32))
        num_col = int(np.frombuffer(images_b[12:16][::-1], dtype=np.int32))

        images = np.frombuffer(images_b[16:], dtype=np.uint8)
        images = images.reshape(num_row, num_col, -1)
        images = np.reshape(images, (-1, num_row, num_col, 1), order='C')

        labels_ix = np.frombuffer(labels_b[8:], dtype=np.uint8)
        labels = np.zeros((labels_ix.size, 10))

        for i in range(labels_ix.size):
            labels[i, labels_ix[i]] = 1
        
        self.read_data(images, labels)

        return


if __name__ == '__main__':
    images_path = 'MNIST_train_images.gz'
    labels_path = 'MNIST_train_labels.gz'
    mnist = MNISTData(images_path, labels_path)
    X, y = mnist.get_batch(50)
#    import cv2
#    cv2.imshow('', X[0,:,:,:])
#    if cv2.waitKey(0) == 27:
#        cv2.destroyAllWindows()
    mnist1, mnist2 = mnist.split(5000)
    print(mnist1.num_samples)
    print(mnist2.num_samples)

