"""

Collection of strings.

"""


class Strings:
    """
    Strings for use in adversarial noise analysis.
    """

    def __init__(
            self,
            fashion: bool = False,
            n_hidden1: int = 5,
            epsilon: float = 0.0,
            gamma: float = 1.0,
        ):
        """
        Initialize object of strings.

        Inputs:
            fashion: bool
                Whether or not to use fashion MNIST or regular MNIST.

            n_hidden1: int
                Number of hidden nodes

            epsilon: float
                Magnitude of adversarial noise

            gamma: float
                Parameter for weight initialization.
        """

        if not fashion:

            self.na_lin_log_path = 'runs/mnist_linear_eps_' \
                + '{0}'.format(epsilon)
            self.na_lin_save_path = 'networks/mnist_eps_' \
                + '{0}'.format(epsilon)
            self.adv_lin_log_path = 'runs/mnist_linear_eps_' \
                + '{0}'.format(epsilon)
            self.adv_lin_save_path = 'networks/mnist_lin_adv_eps_' \
                + '{0}'.format(epsilon)
            self.na_res_log_path = 'runs/mnist_resnet_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.na_res_save_path = 'networks/mnist_resnet_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.adv_res_log_path = 'runs/mnist_resnet_adv_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.adv_res_save_path = 'networks/mnist_resnet_adv_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.na_nn_log_path = 'runs/mnist_nn_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.na_nn_save_path = 'networks/mnist_nn_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.adv_nn_log_path = 'runs/mnist_nn_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.adv_nn_save_path = 'networks/mnist_nn_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.folder = "mnist_output/"

        if fashion:

            self.na_lin_log_path = 'runs/fashion_mnist_linear_nonadv_eps_' \
                + '{0}'.format(epsilon)
            self.na_lin_save_path = 'networks/fashion_mnist_nonadv_eps_' \
                + '{0}'.format(epsilon)
            self.adv_lin_log_path = 'runs/fashion_mnist_linear_eps_' \
                + '{0}'.format(epsilon)
            self.adv_lin_save_path = 'networks/fashion_mnist_lin_adv_eps_' \
                + '{0}'.format(epsilon)
            self.na_res_log_path = 'runs/fashion_mnist_resnet_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.na_res_save_path = 'networks/fashion_mnist_resnet_reg_' \
                + '{0}_hidden_eps_{1}gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.adv_res_log_path = 'runs/fashion_mnist_resnet_adv_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.adv_res_save_path = 'networks/fashion_mnist_resnet_adv_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.na_nn_log_path = 'runs/fashion_mnist_nn_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.na_nn_save_path = 'networks/fashion_mnist_nn_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.adv_nn_log_path = 'runs/fashion_mnist_nn_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.adv_nn_save_path = 'networks/fashion_mnist_nn_reg_' \
                + '{0}_hidden_eps_{1}_gamma_{2}'.format(
                    n_hidden1, epsilon, gamma
                )
            self.folder = "fashion_output/"


        self.results = self.folder \
            + "results_{0}_hidden_eps_{1}_gamma_{2}.pt".format(
                n_hidden1, epsilon, gamma
            )

        self.data = "./data"
        self.start_string = "Beginning training with gamma = " \
            + "{0}, n_hidden1 = {1}.".format(gamma, n_hidden1)
