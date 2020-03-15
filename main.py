import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from models.GAN import GAN
from models.WGAN import WGAN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["GAN", "WGAN"], help="GAN | WGAN")
    parser.add_argument("--dataset", required=True, choices=["MNIST", "CIFAR10"], help="MNIST | CIFAR10")
    parser.add_argument("--batch_size", type=int, default=128, help="Size of the batch used in training.")
    parser.add_argument("--noise_dim", type=int, default=128, help="Dimension of the latent noise.")
    parser.add_argument("--total_epoch", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--visualize", type=bool, default=True,
                        help="Logical indicating whether to visualize the training process and the resulting models.")
    parser.add_argument("--grad_penalty", type=float, default=10.0,
                        help="Penalty factor controlling the strength of the gradient regularization in WGAN-LP.")
    parser.add_argument("--perturb_factor", type=float, default=0.5,
                        help="Pertubation factor for generating samples to compute the gradient penalty in WGAN-LP.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The learning rate of the ADAM optimizer.")
    parser.add_argument("--beta_1", type=float, default=0.5,
                        help="The exponential decay rate for the 1st moment estimates in ADAM.")
    parser.add_argument("--beta_2", type=float, default=0.999,
                        help="The exponential decay rate for the 2nd moment estimates in ADAM.")
    parser.add_argument("--epsilon", type=float, default=1e-7, help="A small constant for numerical stability of ADAM.")
    parser.add_argument("--amsgrad", type=bool, default=False,
                        help="Logical indicating whether to use the AMSGrad variant of ADAM.")
    model_param = parser.parse_args()

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
        plt.savefig("./{}_{}_Example.png".format(model_param.model, model_param.dataset))

        # plot the objective functions
        plt.figure()
        plt.title("Objective Functions of {} (Dataset: {})".format(model_param.model, model_param.dataset))
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.plot(model.d_obj)
        plt.plot(model.g_obj)
        plt.legend(['Discriminator', 'Generator'])
        plt.savefig("./{}_{}_Objective.png".format(model_param.model, model_param.dataset))


if __name__ == "__main__":
    main()
