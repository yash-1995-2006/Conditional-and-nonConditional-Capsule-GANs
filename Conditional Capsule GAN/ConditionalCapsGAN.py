import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from config import cfg
# from utils import softmax
from utils import reduce_sum
from capsLayer import CapsLayer
from capsLayer2 import CapsLayer2
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.neighbors import NearestNeighbors
import utils as utils
from tqdm import tqdm


# G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        y1 = tf.expand_dims(tf.expand_dims(y, axis=1), axis=1)
        x = tf.concat((x, y1), axis=3)
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


# D(x)
def discriminator(input, y, isTrain=True, reuse=False):
    epsilon = 1e-9
    #    if isTrain:
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
            #                print conv1.shape
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
        with tf.variable_scope('Primary_Classifier_layer'):
            digitCaps = CapsLayer(num_outputs=2, vec_len=16, with_routing=True, layer_type='FC')
            caps2 = digitCaps(caps1)  # batch size x 2 x 16 x 1
            v_length = tf.sqrt(reduce_sum(tf.square(caps2), axis=2, keepdims=True) + epsilon)
        with tf.variable_scope('Conditional'):
            condDigitCaps = CapsLayer2(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            condCaps2 = condDigitCaps(caps1)
            cond_v_length = tf.sqrt(reduce_sum(tf.square(condCaps2), axis=2, keepdims=True) + epsilon)
        max_l = tf.square(tf.maximum(0., cfg.m_plus - v_length))  # [batch_size, 10, 1, 1]
        max_r = tf.square(tf.maximum(0., v_length - cfg.m_minus))  # [batch_size, 10, 1, 1]

        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))  # [batch_size, 10]
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))  # [batch_size, 10]
        T_c = Y
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
        margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        cmax_l = tf.square(tf.maximum(0., cfg.m_plus - cond_v_length))
        cmax_r = tf.square(tf.maximum(0., cond_v_length - cfg.m_minus))
        cmax_l = tf.reshape(cmax_l, shape=(cfg.batch_size, -1))
        cmax_r = tf.reshape(cmax_r, shape=(cfg.batch_size, -1))
        cT_c = y
        cL_c = cT_c * cmax_l + cfg.lambda_val * (1 - cT_c) * cmax_r
        class_margin_loss = tf.reduce_mean(tf.reduce_sum(cL_c, axis=1))
        return margin_loss, class_margin_loss


fixed_z_ = np.random.normal(0, 1, (cfg.batch_size, 1, 1, 100))
tempY = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
fixed_y = np.vstack((tempY, tempY, tempY, tempY, tempY, tempY, tempY, tempY, tempY, tempY, tempY, tempY, tempY, tempY))[
          :cfg.batch_size]

y0s = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * cfg.batch_size)
y1s = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * cfg.batch_size)
y2s = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]] * cfg.batch_size)
y3s = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]] * cfg.batch_size)
y4s = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]] * cfg.batch_size)
y5s = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]] * cfg.batch_size)
y6s = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]] * cfg.batch_size)
y7s = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]] * cfg.batch_size)
y8s = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] * cfg.batch_size)
y9s = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] * cfg.batch_size)

# training parameters
batch_size = cfg.batch_size
lr = 0.0002
train_epoch = 2
rotate = False

# load MNIST
train_set, train_setY, val_set, val_setY, test_set, test_setY = utils.load_dataset(name='MNIST')

# variables : input
x = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
ytrain = tf.placeholder(tf.float32, shape=(cfg.batch_size, 10))
z = tf.placeholder(tf.float32, shape=(cfg.batch_size, 1, 1, 100))
ygen = tf.placeholder(tf.float32, shape=(cfg.batch_size, 10))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, ygen, isTrain)
flatG_z = tf.reshape(G_z, [batch_size, -1])
# print G_z.shape

# networks : discriminator
D_real, classD_real = discriminator(x, ytrain, isTrain)
D_fake, classD_fake = discriminator(G_z, ygen, isTrain, reuse=True)

# loss for each network
D_loss = D_real + D_fake + classD_fake + classD_real
G_loss = -D_fake + classD_fake

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
model = 'ConditionalCapsGAN_'
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
if rotate == False:
    rtrain_set, rtrain_setY = train_set, train_setY
else:
    rtrain_set, rtrain_setY = utils.manipulateData(train_set, train_setY)



print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in tqdm(range( len(rtrain_set) // batch_size)):
        # update discriminator
        x_ = rtrain_set[iter * batch_size:(iter + 1) * batch_size]
        y_ = rtrain_setY[iter * batch_size:(iter + 1) * batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, ytrain: y_, ygen: fixed_y, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_, _, flatImage = sess.run([G_loss, G_optim, flatG_z],
                                         {z: z_, x: x_, ytrain: y_, ygen: fixed_y, isTrain: True})
        G_losses.append(loss_g_)
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Images/' + model + str(epoch + 1)

    print('Generating Results')
    print('0s')
    r0 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y0s, save=True, path=fixed_p + '0.png')
    print('1s')
    r1 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y1s, save=True, path=fixed_p + '1.png')
    print('2s')
    r2 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y2s, save=True, path=fixed_p + '2.png')
    print('3s')
    r3 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y3s, save=True, path=fixed_p + '3.png')
    print('4s')
    r4 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y4s, save=True, path=fixed_p + '4.png')
    print('5s')
    r5 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y5s, save=True, path=fixed_p + '5.png')
    print('6s')
    r6 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y6s, save=True, path=fixed_p + '6.png')
    print('7s')
    r7 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y7s, save=True, path=fixed_p + '7.png')
    print('8s')
    r8 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y8s, save=True, path=fixed_p + '8.png')
    print('9s')
    r9 = utils.show_result(sess, (epoch + 1), G_z, flatG_z, z, isTrain, fixed_z_, ygen=ygen, label=y9s, save=True, path=fixed_p + '9.png')

    f0 = open('i0.txt', 'a')
    f1 = open('i1.txt', 'a')
    f2 = open('i2.txt', 'a')
    f3 = open('i3.txt', 'a')
    f4 = open('i4.txt', 'a')
    f5 = open('i5.txt', 'a')
    f6 = open('i6.txt', 'a')
    f7 = open('i7.txt', 'a')
    f8 = open('i8.txt', 'a')
    f9 = open('i9.txt', 'a')

    print('Generating Nearest Neighbours and writing results')
    print('Analysing 0s')
    utils.writeNN(nn=nn, y=rY, tensor=r0, epoch=epoch + 0, file=f0)
    print('Analysing 1s')
    utils.writeNN(nn=nn, y=rY, tensor=r1, epoch=epoch + 1, file=f1)
    print('Analysing 2s')
    utils.writeNN(nn=nn, y=rY, tensor=r2, epoch=epoch + 2, file=f2)
    print('Analysing 3s')
    utils.writeNN(nn=nn, y=rY, tensor=r3, epoch=epoch + 3, file=f3)
    print('Analysing 4s')
    utils.writeNN(nn=nn, y=rY, tensor=r4, epoch=epoch + 4, file=f4)
    print('Analysing 5s')
    utils.writeNN(nn=nn, y=rY, tensor=r5, epoch=epoch + 5, file=f5)
    print('Analysing 6s')
    utils.writeNN(nn=nn, y=rY, tensor=r6, epoch=epoch + 6, file=f6)
    print('Analysing 7s')
    utils.writeNN(nn=nn, y=rY, tensor=r7, epoch=epoch + 7, file=f7)
    print('Analysing 8s')
    utils.writeNN(nn=nn, y=rY, tensor=r8, epoch=epoch + 8, file=f8)
    print('Analysing 9s')
    utils.writeNN(nn=nn, y=rY, tensor=r9, epoch=epoch + 9, file=f9)

    f0.close()
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()
    f8.close()
    f9.close()

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

utils.generateGIFs(train_epoch, root, model, conditional=True, labels=10)
utils.show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
sess.close()
