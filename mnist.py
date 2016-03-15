import os
import struct
import numpy as np
import h5py

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

"""
Source: https://gist.github.com/akesling/5358964
"""

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

if __name__ == '__main__':
    print 'Loading MNIST images...'
    train_image_generator = read(dataset="training", path=".")
    train_image_tuples = [tup for tup in train_image_generator]
    train_image_lbls = np.vstack([tup[0] for tup in train_image_tuples])
    train_images = np.vstack([np.reshape(tup[1], (1, 28 * 28)) for tup in train_image_tuples])

    test_image_generator = read(dataset="testing", path=".")
    test_image_tuples = [tup for tup in test_image_generator]
    test_image_lbls = np.vstack([tup[0] for tup in test_image_tuples])
    test_images = np.vstack([np.reshape(tup[1], (1, 28 * 28)) for tup in test_image_tuples])

    print 'Saving...'
    with h5py.File('MNIST.hdf5', 'w') as f:
        f['train_input'] = train_images
        f['train_output'] = train_image_lbls
        f['test_input'] = test_images
        f['test_output'] = test_image_lbls

