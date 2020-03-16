import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


class GAN:
    def __init__(self, param):
        # load data
        self.dataset = param.get("dataset", -1)

        # choose CNN setup for the dataset
        if self.dataset == "MNIST":
            (self.x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
            self.init_dim = 7
            self.strides = (1, 2, 2)
            self.data_shape = self.x_train.shape + (1,)
        elif self.dataset == "CIFAR10":
            (self.x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
            self.init_dim = 4
            self.strides = (2, 2, 2)
            self.data_shape = self.x_train.shape
        else:
            raise ValueError('Dataset not supported.')

        # get basic inputs
        self.batch_size = param.get("batch_size", 128)
        self.noise_dim = param.get("noise_dim", 128)
        self.total_epoch = param.get("total_epoch", 100)
        self.critic_step = param.get("critic_step", 1)
        self.visualize = param.get("visualize", True)
        self.out_path = param.get("output", os.getcwd())

        # storage for the objectives
        self.batch_num = int(self.data_shape[0] / self.batch_size) + (self.data_shape[0] % self.batch_size != 0)
        self.d_obj = np.zeros(self.total_epoch)
        self.g_obj = np.zeros(self.total_epoch)

        # normalize dataset
        self.x_train = self.x_train.reshape(self.data_shape).astype('float32')
        self.x_train = (self.x_train - 127.5) / 127.5  # Normalize RGB to [-1, 1]
        self.x_train = \
            tf.data.Dataset.from_tensor_slices(self.x_train).shuffle(self.data_shape[0]).batch(self.batch_size)

        # setup optimizers
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=param.get("learning_rate", 1e-4),
                                                    beta_1=param.get("beta_1", 0.5),
                                                    beta_2=param.get("beta_2", 0.999),
                                                    epsilon=param.get("epsilon", 1e-7),
                                                    amsgrad=param.get("amsgrad", False))
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=param.get("learning_rate", 1e-4),
                                                    beta_1=param.get("beta_1", 0.5),
                                                    beta_2=param.get("beta_2", 0.999),
                                                    epsilon=param.get("epsilon", 1e-7),
                                                    amsgrad=param.get("amsgrad", False))

        # setup models
        self.G = self.set_generator()
        self.D = self.set_discriminator()

    def set_generator(self):
        g = tf.keras.Sequential()
        g.add(layers.Dense(self.init_dim * self.init_dim * 256, use_bias=False,
                           input_shape=(self.noise_dim,),
                           kernel_initializer="glorot_uniform"))
        g.add(layers.BatchNormalization())
        g.add(layers.LeakyReLU())
        g.add(layers.Reshape((self.init_dim, self.init_dim, 256)))

        g.add(layers.Conv2DTranspose(128, 5, strides=self.strides[0], padding='same', use_bias=False))
        g.add(layers.BatchNormalization())
        g.add(layers.LeakyReLU())

        g.add(layers.Conv2DTranspose(64, 5, strides=self.strides[1], padding='same', use_bias=False))
        g.add(layers.BatchNormalization())
        g.add(layers.LeakyReLU())

        g.add(layers.Conv2DTranspose(32, 5, strides=self.strides[2], padding='same', use_bias=False))
        g.add(layers.BatchNormalization())
        g.add(layers.LeakyReLU())

        g.add(layers.Conv2DTranspose(self.data_shape[3], 5, padding='same', use_bias=False,
                                     activation='tanh'))

        return g

    def set_discriminator(self):
        d = tf.keras.Sequential()
        d.add(layers.Conv2D(32, kernel_size=5, strides=2, padding='same',
                            input_shape=self.data_shape[1:],
                            kernel_initializer=tf.keras.initializers.glorot_uniform()))
        d.add(layers.LeakyReLU())

        d.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same'))
        d.add(layers.BatchNormalization())
        d.add(layers.LeakyReLU())

        d.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
        d.add(layers.BatchNormalization())
        d.add(layers.LeakyReLU())
        d.add(layers.Dropout(0.5))

        d.add(layers.Conv2D(256, kernel_size=5, strides=2, padding='same'))
        d.add(layers.BatchNormalization())
        d.add(layers.LeakyReLU())

        d.add(layers.Flatten())
        d.add(layers.Dense(2, activation='softmax'))
        d.add(layers.Lambda(lambda x: x[:, 0]))

        return d

    @tf.function
    def train_discriminator(self, x_batch):
        with tf.GradientTape() as D_tape:
            x_gen = self.G(tf.random.uniform([x_batch.shape[0], self.noise_dim]), training=True)
            y_real = self.D(x_batch, training=True)
            y_gen = self.D(x_gen, training=True)

            # compute the objective
            loss_real = tf.math.log(tf.clip_by_value(y_real, 1e-10, 1.0))
            loss_gen = tf.math.log(tf.clip_by_value(tf.math.add(1.0, tf.math.negative(y_gen)), 1e-10, 1.0))
            d_gain = -tf.math.reduce_mean(loss_real + loss_gen)

            # update the discriminator
            d_grad = D_tape.gradient(d_gain, self.D.trainable_variables)
            self.D_optimizer.apply_gradients(zip(d_grad, self.D.trainable_variables))
        return d_gain

    @tf.function
    def train_generator(self, x_batch_size):
        with tf.GradientTape() as G_tape:
            x_gen = self.G(tf.random.uniform([x_batch_size, self.noise_dim]), training=True)
            y_gen = self.D(x_gen, training=True)

            # update the generator
            g_gain = -tf.math.reduce_mean(tf.math.log(tf.clip_by_value(y_gen, 1e-10, 1.0)))
            g_grad = G_tape.gradient(g_gain, self.G.trainable_variables)
            self.G_optimizer.apply_gradients(zip(g_grad, self.G.trainable_variables))
        return g_gain

    def train(self):
        if self.visualize:
            # Seed for checking training progress
            vis_seed = tf.random.uniform([16, self.noise_dim])

        # Record current time and start training
        print("Training...")
        ts_start = tf.timestamp()
        for t in range(self.total_epoch):
            for b in self.x_train:
                for k in range(self.critic_step):
                    self.d_obj[t] -= self.train_discriminator(b)
                self.g_obj[t] -= self.train_generator(b.shape[0])
            self.d_obj[t] /= self.critic_step * self.batch_num
            self.g_obj[t] /= self.batch_num

            # Print time
            print("Time used for epoch {} are {:0.2f} seconds.".format(t + 1, tf.timestamp() - ts_start))

            # Check current generator
            if self.visualize:
                vis_gen = self.G(vis_seed, training=False)
                fig = plt.figure(figsize=(4, 4))
                plt.suptitle('Epoch: {:03d}'.format(t + 1))
                for i in range(vis_gen.shape[0]):
                    plt.subplot(4, 4, i + 1)
                    if self.data_shape[3] == 1:
                        plt.imshow(vis_gen[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
                    else:
                        plt.imshow((vis_gen[i, :, :] + 1) / 2)
                    plt.axis('off')
                plt.savefig(os.path.join(self.out_path, "GAN_{}_Epoch_{:03d}.png".format(self.dataset, t + 1)))
                plt.clf()
                plt.close(fig)
        print("Done! {:0.2f} seconds have passed.".format(tf.timestamp() - ts_start))
