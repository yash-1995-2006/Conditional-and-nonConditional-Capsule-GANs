import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from config import cfg
import utilities as utils
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from tqdm import tqdm
import generators as gen

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        return gen.nonConditionalGenerator(x, isTrain=isTrain)

# D(x)
def discriminator(input, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        conv1 = tcl.conv2d(input, 64, 5, 2, activation_fn=tf.identity,
                           weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
        conv1 = utils.lrelu(conv1)
        conv2 = tcl.conv2d(conv1, 128, 5, 2, activation_fn=tf.identity,
                           weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
        conv2 = utils.lrelu(conv2)
        conv3 = tcl.conv2d(conv2, 256, 5, 2, activation_fn=tf.identity,
                           weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
        conv3 = utils.lrelu(conv3)
        conv4 = tcl.conv2d(conv3, 512, 5, 2, activation_fn=tf.identity,
                           weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
        conv4 = utils.lrelu(conv4)
        conv5 = tcl.conv2d(conv4, 1, 4, 1, activation_fn=tf.identity,
                           weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
        return conv5



# training parameters
batch_size = cfg.batch_size
lr = 0.0002
train_epoch = 2
rotate = False

#load dataset
train_set, train_setY, val_set, val_setY, test_set, test_setY = utils.load_dataset(name='MNIST')

# variables : input
x = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
z = tf.placeholder(tf.float32, shape=(cfg.batch_size, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)
fixed_z_ = np.random.normal(0, 1, (cfg.batch_size, 1, 1, 100))

# networks : generator
G_z = generator(z, isTrain)
flatG_z = tf.reshape(G_z, [batch_size, -1])

# networks : discriminator
D_real = discriminator(x, isTrain)
D_fake = discriminator(G_z, isTrain, reuse=True)

#Losses
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = tf.reduce_mean(D_fake)

epsilon = tf.random_uniform([], 0.0, 1.0)
x_hat = x*epsilon + (1-epsilon)*G_z
d_hat = discriminator(x_hat, batch_size, reuse=True)
gradients = tf.gradients(d_hat, x_hat)[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
D_loss += gradient_penalty

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print('Saving initial model')
saver.save(sess, './model')

# results save folder
root = 'Results/'
model = 'Improved_Wasserstein_DCGAN'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Images'):
    os.mkdir(root + 'Images')

#store trainig hitory
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))

print('Stacking initial dataset')
X, Y = utils.stackDataset(train_set,train_setY, val_set, val_setY, test_set, test_setY)

print('Flipping Data and then restacking')
rX, rY = utils.flipRestack(X, Y)

print('Reshaping Data')
X, Y = utils.reshapeForNN(X, Y)
rX, rY = utils.reshapeForNN(X, Y)

print('Fitting to Nearest Neighbours model')
nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
if rotate == True:
    nn.fit(rX)
else:
    nn.fit(X)

print('Setting up training data')
rtrain_set, rtrain_setY = train_set, train_setY
if rotate == True:
    rtrain_set, rtrain_setY = utils.manipulateData(train_set, train_setY, type='flip')

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    #for iter in tqdm(range(rtrain_set.shape[0] // batch_size)):
    for iter in tqdm(range(3)):
        # update discriminator
        x_ = rtrain_set[iter * batch_size:(iter + 1) * batch_size]
        y_ = rtrain_setY[iter * batch_size:(iter + 1) * batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_, _, flatImage = sess.run([G_loss, G_optim, flatG_z],
                                         {z: z_, x: x_, isTrain: True})
        G_losses.append(loss_g_)
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Images/' + model + str(epoch + 1)

    print('Generating Results')
    flatImages = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_ = fixed_z_, save=True, path=fixed_p + '.png')
    print('Generating Nearest Neighbours and writing results')
    print('Analysing 0s')
    f0 = open('NN_Distances.txt', 'a')
    utils.writeNN(nn=nn, y = rY, tensor=flatImages, epoch=epoch + 0, file=f0)
    f0.close()

    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    print('Saving Model Epoch: ', epoch + 1)
    saver.save(sess, './model', write_meta_graph=False)
    print('Model Saved\n')

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

utils.generateGIFs(train_epoch, root, model, conditional=False, labels=0)

utils.show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
sess.close()
