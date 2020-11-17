"""

Evaluate the robust models.

"""


import torch
import torchvision
import torchvision.transforms as transforms
from evaluation_functions import evaluate_all
from strings import Strings


def main():
    """
    Evaluate the robust models in both the non-adversarial and
    adversarial settings.
    """

    strings = Strings()

    ## Parameters for the runs.
    fashion_list = [True, False]
    n_epochs = 5
    batch_size = 10
    epsilon = 0.05
    gamma_list = [1., 5., 10.]
    n_hidden1_list = [150, 200, 300, 400, 500]
    n_inputs = 28 * 28
    n_outputs = 10

    ## Loading Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    mnist_trainset = torchvision.datasets.MNIST(
        strings.data,
        download=False,
        train=True,
        transform=transform
    )
    mnist_testset = torchvision.datasets.MNIST(
        strings.data,
        download=False,
        train=False,
        transform=transform
    )

    fashion_trainset = torchvision.datasets.FashionMNIST(
        strings.data,
        download=False,
        train=True,
        transform=transform
    )
    fashion_testset = torchvision.datasets.FashionMNIST(
        strings.data,
        download=False,
        train=False,
        transform=transform
    )

    # Dataloaders
    mnist_trainloader = torch.utils.data.DataLoader(
        mnist_trainset,
        batch_size=batch_size,
        shuffle=True
    )
    mnist_testloader = torch.utils.data.DataLoader(
        mnist_testset,
        batch_size=batch_size,
        shuffle=False
    )

    fashion_trainloader = torch.utils.data.DataLoader(
        fashion_trainset,
        batch_size=batch_size,
        shuffle=True
    )
    fashion_testloader = torch.utils.data.DataLoader(
        fashion_testset,
        batch_size=batch_size,
        shuffle=False
    )

    # Run experiments
    for fashion in fashion_list:
        for gamma in gamma_list:
            for n_hidden1 in n_hidden1_list:


                if not fashion:
                    trainloader = mnist_trainloader
                    testloader = mnist_testloader

                else:
                    trainloader = fashion_trainloader
                    testloader = fashion_testloader

                strings = Strings(fashion, n_hidden1, epsilon, gamma)
                print(strings.start_string)

                evaluate_all(
                    n_hidden1,
                    gamma,
                    n_inputs,
                    n_outputs,
                    epsilon,
                    n_epochs,
                    trainloader,
                    testloader,
                    strings.na_lin_log_path,
                    strings.na_lin_save_path,
                    strings.adv_lin_log_path,
                    strings.adv_lin_save_path,
                    strings.na_res_log_path,
                    strings.na_res_save_path,
                    strings.adv_res_log_path,
                    strings.adv_res_save_path,
                    strings.na_nn_log_path,
                    strings.na_nn_save_path,
                    strings.adv_nn_log_path,
                    strings.adv_nn_save_path,
                    strings.results
                )


if __name__ == "__main__":
    main()
