import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


class WGAN:
    def __init__(self, param):
        # load data
        self.dataset = param.get("dataset", -1)

        # choose CNN setup for each dataset
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

        # set regularization parameters
        self.grad_penalty = param.get("grad_penalty", 10.0)
        self.perturb_factor = param.get("perturb_factor", 0.5)

        # normalize dataset
        self.x_train = self.x_train.reshape(self.data_shape).astype('float32')
        self.x_train = (self.x_train - 127.5) / 127.5  # Normalize RGB to [-1, 1]
        self.x_train = \
            tf.data.Dataset.from_tensor_slices(self.x_train).shuffle(self.data_shape[0]).batch(self.batch_size)

        # setup optimizers
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=param.get("learning_rate", 2e-4),
                                                    beta_1=param.get("beta_1", 0.5),
                                                    beta_2=param.get("beta_2", 0.999),
                                                    epsilon=param.get("epsilon", 1e-7),
                                                    amsgrad=param.get("amsgrad", False))
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=param.get("learning_rate", 5e-5),
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
        d.add(layers.LayerNormalization())
        d.add(layers.LeakyReLU())

        d.add(layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
        d.add(layers.LayerNormalization())
        d.add(layers.LeakyReLU())

        d.add(layers.Conv2D(256, kernel_size=5, strides=2, padding='same'))
        d.add(layers.LayerNormalization())
        d.add(layers.LeakyReLU())

        d.add(layers.Flatten())
        d.add(layers.Dense(1))

        return d

    @tf.function
    def lipschitz_penalty(self, x, x_hat):
        # DRAGAN-like sampling scheme
        x_join = tf.concat([x, x_hat], axis=0)
        _, batch_var = tf.nn.moments(x_join, axes=[0, 1, 2, 3])
        delta = self.perturb_factor * batch_var * tf.random.uniform([tf.shape(x_join)[0], 1, 1, 1])
        epsilon = tf.random.uniform([tf.shape(x_join)[0], 1, 1, 1])
        x_tilde = x_join + (1 - epsilon) * delta

        # compute gradient penalty
        with tf.GradientTape() as D_tape:
            D_tape.watch(x_tilde)
            y_tilde = self.D(x_tilde, training=False)
        d_grad = D_tape.gradient(y_tilde, x_tilde)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(d_grad), axis=[1, 2, 3]))

        return tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.0)))

    @tf.function
    def train_discriminator(self, x_batch):
        with tf.GradientTape() as D_tape:
            # sample data
            x_gen = self.G(tf.random.uniform([x_batch.shape[0], self.noise_dim]), training=True)

            # scoring with the discriminator
            y_real = self.D(x_batch, training=True)
            y_gen = self.D(x_gen, training=True)

            # compute the objective
            d_obj = tf.math.reduce_mean(y_gen) - tf.math.reduce_mean(y_real)
            d_obj_pen = d_obj + self.grad_penalty * self.lipschitz_penalty(x_batch, x_gen)

            # update the discriminator
            d_grad = D_tape.gradient(d_obj_pen, self.D.trainable_variables)
            self.D_optimizer.apply_gradients(zip(d_grad, self.D.trainable_variables))
        return d_obj

    @tf.function
    def train_generator(self, x_batch_size):
        with tf.GradientTape() as G_tape:
            x_gen = self.G(tf.random.uniform([x_batch_size, self.noise_dim]), training=True)
            y_gen = self.D(x_gen, training=True)

            # compute the objective
            g_obj = -tf.math.reduce_mean(y_gen)

            # update the generator
            g_grad = G_tape.gradient(g_obj, self.G.trainable_variables)
            self.G_optimizer.apply_gradients(zip(g_grad, self.G.trainable_variables))
        return g_obj

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
                self.g_obj[t] += self.train_generator(b.shape[0])
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
                plt.savefig(os.path.join(self.out_path, "WGAN_{}_Epoch_{:03d}.png".format(self.dataset, t + 1)))
                plt.clf()
                plt.close(fig)
        print("Done! {:0.2f} seconds have passed.".format(tf.timestamp() - ts_start))
