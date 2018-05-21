import os
import scipy
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import itertools
import imageio
import matplotlib.pyplot as plt

# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)



def load_dataset(name='MNIST'):
    if name == 'MNIST':
        data = input_data.read_data_sets("MNIST_data/", one_hot = True, reshape = [])
    elif name == 'Fashion':
        data = input_data.read_data_sets('data/fashion',
                                  source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
                                  one_hot=True, reshape=[])

    # normalization; range: -1 ~ 1
    train_set = data.train.images
    train_set = (train_set - 0.5) / 0.5
    train_setY = data.train.labels
    test_set = data.test.images
    test_set = (test_set - 0.5) / 0.5
    test_setY = data.test.labels
    val_set = data.validation.images
    val_set = (val_set - 0.5) / 0.5
    val_setY = data.validation.labels

    return train_set, train_setY, val_set, val_setY, test_set, test_setY



def show_result(sess, num_epoch, G_z, flatG_z, z, isTrain, fixed_z_, ygen=None, label=[], show=False, save=False, path='result.png'):
    if label == []:
        test_images, flat = sess.run([G_z, flatG_z], {z: fixed_z_, isTrain: False})
    else:
        test_images, flat = sess.run([G_z, flatG_z], {z: fixed_z_, ygen: label, isTrain: False})
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(size_figure_grid * size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()
    return flat[:(size_figure_grid * size_figure_grid)]


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))
    y1 = hist['D_losses']
    y2 = hist['G_losses']
    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def writeNN(nn, y, tensor, epoch, file):
    '''
    :param nn: initialized sklearn nearest neighbour object
    :param y: data labels
    :param tensor: generated flattened samples
    :param epoch: epoch number
    :param file: file handle to store in
    :return: None
    '''
    distance, index = nn.kneighbors(tensor)
    labels = y[index]
    file.write('\n\nEpoch: ' + str(epoch+1) + ' Average Distance: ' + str(np.mean(distance)) + '\n')
    print('Average Distance: ', np.mean(distance))
    for l in range(len(tensor)):
        file.write('Closest label: ' + str(labels[l]) + ' distance: ' + str(distance[l]) + '\n')


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def conv_cond_concat(x, y):
   x_shapes = x.get_shape()
   y_shapes = y.get_shape()
   return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def stackDataset(train_set, train_setY, val_set, val_setY, test_set, test_setY):
    X, Y = train_set, train_setY
    X, Y = np.vstack((X, test_set)), np.vstack((Y, test_setY))
    X, Y = np.vstack((X, val_set)), np.vstack((Y, val_setY))

    return X,Y

def reshapeForNN(X, Y):
    return X.reshape((len(X), -1)), Y.reshape((len(Y), -1))

def flipRestack(X, Y):
    rX, rY = X, Y
    rX, rY = np.vstack((rX, np.flip(X, 2))), np.vstack((rY, Y))
    return rX, rY

def rotateRestack(X, Y):
    rX, rY = X, Y
    rX, rY = np.vstack((rX, np.rot90(X, 1, (1, 2)))), np.vstack((rY, Y))
    rX, rY = np.vstack((rX, np.rot90(X, 2, (1, 2)))), np.vstack((rY, Y))
    rX, rY = np.vstack((rX, np.rot90(X, 3, (1, 2)))), np.vstack((rY, Y))
    return rX, rY

def manipulateData(x, y, type='flip'):
    rtrain_set, rtrain_setY = x, y
    if type == 'flip':
        rtrain_set, rtrain_setY = np.vstack((rtrain_set, np.flip(x, 2))), np.vstack((rtrain_setY, y))
    if type == 'rotate':
        rtrain_set, rtrain_setY = np.vstack((rtrain_set, np.rot90(x, 1, (1, 2)))), np.vstack(
            (rtrain_setY, y))
        rtrain_set, rtrain_setY = np.vstack((rtrain_set, np.rot90(x, 2, (1, 2)))), np.vstack(
            (rtrain_setY, y))
        rtrain_set, rtrain_setY = np.vstack((rtrain_set, np.rot90(x, 3, (1, 2)))), np.vstack(
            (rtrain_setY, y))
    indices = np.random.permutation(rtrain_setY.shape[0])
    rtrain_set, rtrain_setY = rtrain_set[indices], rtrain_setY[indices]
    return rtrain_set, rtrain_setY

def generateGIFs(epoch, root, model, conditional=False, labels=0):
    if conditional == False:
        images = []
        for e in range(epoch):
            img_name = root + 'Images/' + model + str(e + 1) + '.png'
            images.append(imageio.imread(img_name))
        imageio.mimsave(root + model + 'generation_animation' + '.gif', images, fps=5)
    elif conditional == True:
        for i in range(labels):
            images = []
            for e in range(epoch):
                img_name = root + 'Images/' + model + str(e + 1) + str(i) + '.png'
                images.append(imageio.imread(img_name))
            imageio.mimsave(root + model + 'generation_animation' + str(i) + '.gif', images, fps=5)