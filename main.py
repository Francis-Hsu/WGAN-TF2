import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.GAN import GAN
from models.WGAN import WGAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["GAN", "WGAN"], required=True, help="GAN | WGAN")
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], required=True, help="MNIST | CIFAR10")
    parser.add_argument("--output", default=os.getcwd(), type=str, help="Directory to save the output (if any).")
    parser.add_argument("--batch_size", default=128, type=int, help="Size of the batch used in training.")
    parser.add_argument("--noise_dim", default=128, type=int, help="Dimension of the latent noise.")
    parser.add_argument("--total_epoch", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--critic_step", default=1, type=int,
                        help="The number of steps to apply to the discriminator.")
    parser.add_argument("--visualize", default=True, type=bool,
                        help="Logical indicating whether to visualize the training process and the resulting models.")
    parser.add_argument("--grad_penalty", default=10.0, type=float,
                        help="Penalty controlling the strength of the gradient regularization in WGAN-LP.")
    parser.add_argument("--perturb_factor", default=0.5, type=float,
                        help="Factor controlling the standard deviation of perturbation "
                             "for generating samples to compute the gradient penalty in WGAN-LP.")

    # setup the discriminator optimizer
    parser.add_argument("--learning_rate_d", default=1e-4, type=float,
                        help="The learning rates of ADAM (discriminator).")
    parser.add_argument("--beta_1_d", default=0.2, type=float,
                        help="The exponential decay rates for the 1st moment estimates in ADAM (discriminator).")
    parser.add_argument("--beta_2_d", default=0.999, type=float,
                        help="The exponential decay rates for the 2nd moment estimates in ADAM (discriminator).")
    parser.add_argument("--epsilon_d", default=1e-7, type=float,
                        help="Small constants for numerical stability of ADAM (discriminator).")
    parser.add_argument("--amsgrad_d", default=False, type=bool,
                        help="Logical indicating whether to use the AMSGrad variant of ADAM (discriminator).")

    # setup the generator optimizer
    parser.add_argument("--learning_rate_g", default=1e-4, type=float,
                        help="The learning rates of ADAM (generator).")
    parser.add_argument("--beta_1_g", default=0.5, type=float,
                        help="The exponential decay rates for the 1st moment estimates in ADAM (generator).")
    parser.add_argument("--beta_2_g", default=0.999, type=float,
                        help="The exponential decay rates for the 2nd moment estimates in ADAM (generator).")
    parser.add_argument("--epsilon_g", default=1e-7,
                        type=float, help="Small constants for numerical stability of ADAM (generator).")
    parser.add_argument("--amsgrad_g", default=False, type=bool,
                        help="Logical indicating whether to use the AMSGrad variant of ADAM (generator).")
    model_param = parser.parse_args()

    if not os.path.exists(model_param.output) or not os.path.isdir(model_param.output):
        raise OSError("Output path does not exist or is not a directory.")
    model_param.output = os.path.normpath(model_param.output)

    if model_param.model == "GAN":
        model = GAN(vars(model_param))
    else:
        model = WGAN(vars(model_param))
    model.train()

    # randomly generate 100 samples
    if model_param.visualize:
        vis_seed = tf.random.uniform([100, model.noise_dim])
        vis_gen = model.G(vis_seed, training=False)
        if model_param.dataset == "MNIST":
            plt.figure(figsize=(3.45, 3.45))
        else:
            plt.figure(figsize=(3.85, 3.85))
        for i in range(vis_gen.shape[0]):
            x_pos = i % 10
            y_pos = int(i / 10)
            if model_param.dataset == "MNIST":
                plt.figimage(vis_gen[i, :, :, 0] * 127.5 + 127.5,
                             10 + x_pos * (28 + 5), 10 + y_pos * (28 + 5), cmap='gray')
            else:
                plt.figimage((vis_gen[i, :, :] + 1) / 2,
                             10 + x_pos * (32 + 5), 10 + y_pos * (32 + 5))
            plt.axis('off')
        plt.savefig(os.path.join(model_param.output,
                                 "{}_{}_Example.png".format(model_param.model, model_param.dataset)))

        # plot median value of the objective functions
        plt.figure()
        plt.title("Objective Functions of {} (Dataset: {})".format(model_param.model, model_param.dataset))
        plt.xlabel("Epoch")
        plt.ylabel("Median Value")
        plt.plot(range(1, 1 + model_param.total_epoch), np.median(model.d_obj, axis=[-0, -1]))
        plt.plot(range(1, 1 + model_param.total_epoch), np.median(model.g_obj, axis=[-0]))
        plt.legend(['Discriminator', 'Generator'])
        plt.savefig(os.path.join(model_param.output,
                                 "{}_{}_Objective.png".format(model_param.model, model_param.dataset)))


if __name__ == "__main__":
    main()
