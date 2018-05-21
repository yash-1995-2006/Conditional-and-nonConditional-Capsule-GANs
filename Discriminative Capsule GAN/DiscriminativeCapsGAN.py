import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from config import cfg
from utils import reduce_sum
from capsLayer import CapsLayer
import utils as utils
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm





def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = utils.lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [3, 3], strides=(1, 1), padding='valid')
        lrelu2 = utils.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [2, 2], strides=(1, 1), padding='valid')
        lrelu3 = utils.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = utils.lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)
        return o


def discriminator(input, isTrain=True, reuse=False):
    epsilon = 1e-9
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            labels = tf.constant(0, shape=[cfg.batch_size, ])
        else:
            labels = tf.constant(1, shape=[cfg.batch_size, ])
        Y = tf.one_hot(labels, depth=2, axis=1, dtype=tf.float32)
        if reuse:
            scope.reuse_variables()
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(input, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=2, vec_len=16, with_routing=True, layer_type='FC')
            caps2 = digitCaps(caps1)  # batch size x 2 x 16 x 1
            v_length = tf.sqrt(reduce_sum(tf.square(caps2), axis=2, keepdims=True) + epsilon)

        max_l = tf.square(tf.maximum(0., cfg.m_plus - v_length))
        max_r = tf.square(tf.maximum(0., v_length - cfg.m_minus))
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))
        T_c = Y
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
        margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        return margin_loss


fixed_z_ = np.random.normal(0, 1, (cfg.batch_size, 1, 1, 100))

# training parameters
batch_size = cfg.batch_size
lr = 0.0002
train_epoch = 10
rotate = False

# load MNIST
train_set, train_setY, val_set, val_setY, test_set, test_setY = utils.load_dataset(name='MNIST')

# variables : input
x = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
z = tf.placeholder(tf.float32, shape=(cfg.batch_size, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, isTrain)
flatG_z = tf.reshape(G_z, [batch_size, -1])
# print G_z.shape

# networks : discriminator
D_real = discriminator(x, isTrain)
D_fake = discriminator(G_z, isTrain, reuse=True)

# loss for each network
D_loss = D_real + D_fake
G_loss = -D_fake

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
model = 'DiscriminativeCapsGAN'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Images'):
    os.mkdir(root + 'Images')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))

print('Stacking initial dataset')
X, Y = utils.stackDataset(train_set, train_setY, val_set, val_setY, test_set, test_setY)

print('Rotating Data and then restacking')
rX, rY = utils.rotateRestack(X, Y)

print('Reshaping Data')
X, Y = utils.reshapeForNN(X, Y)
rX, rY = utils.reshapeForNN(rX, rY)

print('Fitting to Nearest Neighbours model')
nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
if rotate == True:
    nn.fit(rX)
else:
    nn.fit(X)

print('Setting up training data')
rtrain_set, rtrain_setY = train_set, train_setY
if rotate == True:
    rtrain_set, rtrain_setY = utils.manipulateData(rtrain_set, rtrain_setY)

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in tqdm(range(len(rtrain_set) // batch_size)):
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

    r0 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, save=True, path=fixed_p + '.png')

    f0 = open('NNDistances.txt', 'a')
    print('Generating Nearest Neighbours and writing results')
    print('Analysing')
    utils.writeNN(nn=nn, y=rY, tensor=r0, epoch=epoch, file=f0)
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

images = []
for e in range(train_epoch):
    img_name = root + 'Images/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation' + '.gif', images, fps=5)

utils.show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
sess.close()
