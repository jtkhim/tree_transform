"""

Functions for evaluation adversarial networks.

"""


from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from adversarial_networks import AdvLinearNet, AdvOneLayer, AdvResNet


def one_hot(
        labels: torch.Tensor,
        n_outputs: int,
    ) -> torch.Tensor:
    """
    Returns a +1/-1 one-hot encoding, needed for adversary.

    Inputs:
        labels: torch.Tensor
            Tensor of labels.

        n_outputs: int
            Number of outputs to consider.

    Outputs:
        one_hot: torch.Tensor
            One-hot encoding of labels.
    """

    batch_size = list(labels.shape)[0]

    one_hot_vector = -torch.ones((batch_size, n_outputs))
    for i, label in enumerate(labels):
        one_hot_vector[i, label] = 1.

    return one_hot_vector


def optimize_and_log(
        adv_net: nn.Module,
        n_epochs: int,
        trainloader: DataLoader,
        n_inputs: int,
        n_outputs: int,
        log_path: str,
        save_path: str,
        adv: bool = False,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        save: bool = False
    ) -> float:
    """
    Optimize the network and log the results. Note the adv_net is modified.

    Inputs:
        adv_net: nn.Module
            An adversarial network. AdvLinearNet, AdvOneLayer, AdvResNet.
            Modified by training.

        n_epochs: int
            Number of epochs to train.

        trainloader: DataLoader
            PyTorch DataLoader for training data.

        n_inputs: int
            Number of inputs.

        n_outputs: int
            Number of outputs.

        log_path: str
            Path for logging results.

        save_path: str
            Path for saving model.

        adv: bool
            Whether or not optimization should be done with
            adversarial noise.

        learning_rate: float
            Learning rate for gradient descent algorithm.

        momentum: float
            Momentum parameter.

        save: bool
            Whether or not to save the model

    Outputs:
        train_time: float
            Amount of time in seconds required for training.

        logs:
            Logged results.

        model:
            Model saved to path.
    """

    # Optimization Parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        adv_net.parameters(),
        lr=learning_rate,
        momentum=momentum
    )
    epsilon = adv_net.get_epsilon()

    # Logging
    writer = SummaryWriter(log_path)
    running_loss = 0.0
    begin_time = time()

    for epoch in range(n_epochs):

        print("Beginning epoch {0}.".format(epoch + 1))
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            one_hot_targets = one_hot(labels, n_outputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            current_batch_size = inputs.shape[0]
            outputs = adv_net(
                inputs.reshape([current_batch_size, n_inputs]),
                one_hot_targets,
                adv=adv
            )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update progress.
            if i % 1000 == 0:
                print("Completed batch {}.".format(i))
                writer.add_scalar(
                    'training loss, eps = {}'.format(epsilon),
                    running_loss / 1000.,
                    epoch * len(trainloader) + i
                )
                running_loss = 0.0

        # Model saving.
        if save:
            torch.save(adv_net.state_dict(), save_path)

    writer.close()
    if adv:
        print('Finished training adversarially')
    else:
        print('Finished training non-adversarially')
    return time() - begin_time


def evaluate_classifier(
        net: nn.Module,
        n_inputs: int,
        n_outputs: int,
        testloader: DataLoader,
        adv: bool = False,
    ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates the classifier.

    Inputs:
        net: nn.Module
            Adversarial network for evaluation.

        n_inputs: int
            Number of inputs.

        n_outputs: int
            Number of outputs.

        testloader: DataLoader
            DataLoader for test data.

        adv: bool
            Whether or not there is adversarial noise.

    Outputs:
        test_labels: torch.Tensor
            Test labels

        test_probs: torch.Tensor
            Test probabilities

        test_preds: torch.Tensor
            Test predictions
    """

    class_labels = []
    class_probs = []
    class_preds = []

    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            one_hot_labels = one_hot(labels, n_outputs)
            current_batch_size = inputs.shape[0]
            outputs = net(
                inputs.reshape([current_batch_size, n_inputs]),
                one_hot_labels,
                adv=adv
            )
            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
            _, class_preds_batch = torch.max(outputs, 1)

            class_labels.append(labels)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
            criterion = nn.CrossEntropyLoss()

            if i % 1000 == 0:
                print("Completed validation on batch {}.".format(i))
                cross_entropy_loss = criterion(outputs, labels)
                print("Cross entropy loss is {0}.".format(cross_entropy_loss))

    test_labels = torch.cat(class_labels)
    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    return([test_labels, test_probs, test_preds])


def train_and_test(
        net: nn.Module,
        start_string: str,
        n_epochs: int,
        adv_train: bool,
        n_inputs: int,
        n_outputs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        log_path: str,
        save_path: str,
    ) -> torch.Tensor:
    """
    Train and test an network. Training is possibly adversarial;
    testing is both adversarial and non-adversarial.

    Inputs:
        net: nn.Module
            Neural network to test.

        start_string: str
            String to display upon beginning.

        n_epochs: int
            Number of epochs to train.

        adv_train: bool
            Whether to train adversarially or not.

        n_inputs: int
            Number of inputs.

        n_outputs: int
            Number of outputs.

        trainloader:
            DataLoader for training data

        testloader:
            DataLoader of test data

        log_path: str
            Path to log results.

        save_path: str
            Path to save model.

    Outputs:
        na_misclass: float
            Misclassification rate for non-adversarial classifier.

        adv_misclass: float
            Adversarial misclassification rate.

        train_time: float
            Train time in seconds.
    """

    print(start_string)
    train_time = optimize_and_log(
        net, n_epochs, trainloader,
        n_inputs, n_outputs,
        log_path,
        save_path=save_path,
        save=True,
        adv=adv_train
    )

    # Regular testing
    na_labels, _, na_preds = evaluate_classifier(
        net, n_inputs, n_outputs, testloader, adv=False
    )
    na_misclass = np.mean(na_labels.numpy() != na_preds.numpy())

    # Adversarial testing
    adv_labels, _, adv_preds = evaluate_classifier(
        net, n_inputs, n_outputs, testloader, adv=True
    )
    adv_misclass = np.mean(adv_labels.numpy() != adv_preds.numpy())

    return torch.tensor([na_misclass, adv_misclass, train_time])


def evaluate_na_adv(
        na_net: nn.Module,
        na_start_string: str,
        adv_net: nn.Module,
        adv_start_string: str,
        n_inputs: int,
        n_outputs: int,
        n_epochs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        na_log_path: str,
        na_save_path: str,
        adv_log_path: str,
        adv_save_path: str,
    ) -> torch.Tensor:
    """
    Evaluate non-adversarial classifier and adversarial classifier.

    Inputs:
        na_net: nn.Module
            Non-adversarial neural network.

        na_start_string: str
            Starting string for non-adversarial network.

        adv_net: nn.Module
            Adversarial neural network.

        n_inputs: int
            Number of inputs

        n_outputs: int
            Number of outputs

        trainloader: DataLoader
            DataLoader for training data

        testloader: DataLoader
            DataLoader for test data

        na_log_path: str
            Non-adversarial network results log path.

        na_save_path: str
            Non-adversarial network model save path.

        adv_log_path: str
            Adversarial network results log path.

        adv_save_path: str
            Adversarial network model save path.

    Outputs:
        results: torch.Tensor
            Table containing results.
            The first row contains non-adversarial network results.
            The second row contains results for the adversarial network.
    """

    results = torch.zeros((2, 3))

    results[0, :] = train_and_test(
        na_net, na_start_string,
        n_epochs, False,
        n_inputs, n_outputs,
        trainloader, testloader,
        na_log_path, na_save_path
    )
    results[1, :] = train_and_test(
        adv_net, adv_start_string,
        n_epochs, True,
        n_inputs, n_outputs,
        trainloader, testloader,
        adv_log_path, adv_save_path
    )

    return results


def evaluate_linear(
        n_inputs: int,
        n_outputs: int,
        epsilon: float,
        n_epochs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        na_log_path: str,
        na_save_path: str,
        adv_log_path: str,
        adv_save_path: str,
    ) -> torch.Tensor:
    """
    Create and evaluate a linear network.

    Inputs:
        n_inputs: int
            Number of inputs.

        n_outputs: int
            Number of outputs.

        epsilon: float
            Adversarial noise magnitude.

        n_epochs: int
            Number of training epochs.

        trainloader: DataLoader
            DataLoader for training data.

        testloader: DataLoader
            DataLoader for test data.

        na_log_path: str
            Non-adversarial log path.

        na_save_path: str
            Non-adversarial model save path.

        adv_log_path: str
            Adversarial log path.

        adv_save_path: str
            Adversarial model save path.

    Outputs:
        results: torch.Tensor
            Table containing results.
            The first row contains non-adversarial network results.
            The second row contains results for the adversarial network.
    """

    na_lin_cls = AdvLinearNet(n_inputs, n_outputs, epsilon=epsilon)
    adv_lin_cls = AdvLinearNet(n_inputs, n_outputs, epsilon=epsilon)
    na_start_string = "Starting regular training of the linear classifier."
    adv_start_string = "Starting adversarial training of the linear classifier."

    return evaluate_na_adv(
        na_lin_cls, na_start_string, adv_lin_cls, adv_start_string,
        n_inputs, n_outputs, n_epochs,
        trainloader, testloader,
        na_log_path, na_save_path, adv_log_path, adv_save_path
    )


def evaluate_resnet(
        n_hidden1: int,
        gamma: float,
        n_inputs: int,
        n_outputs: int,
        epsilon: float,
        n_epochs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        na_log_path: str,
        na_save_path: str,
        adv_log_path: str,
        adv_save_path: str,
    ) -> torch.Tensor:
    """
    Create and evaluate a resnet.

    Inputs:
        n_hidden1: int
            Number of hidden layers.

        gamma: float
            Initialization parameter.

        n_inputs: int
            Number of inputs.

        n_outputs: int
            Number of outputs.

        epsilon: float
            Adversarial noise magnitude.

        n_epochs: int
            Number of epochs.

        trainloader: DataLoader
            DataLoader for train data.

        testloader: DataLoader
            DataLoader for test data.

        na_log_path: str
            Non-adversarial model log path.

        na_save_path: str
            Non-adversarial model save path.

        adv_log_path: str
            Adversarial model results log path.

        adv_save_path: str
            Adversarial model save path.

    Outputs:
        results: torch.Tensor
            Table containing results.
            The first row contains non-adversarial network results.
            The second row contains results for the adversarial network.
    """

    na_res_cls = AdvResNet(
        n_inputs, n_outputs, n_hidden1, epsilon=epsilon, gamma=gamma
    )
    adv_res_cls = AdvResNet(
        n_inputs, n_outputs, n_hidden1, epsilon=epsilon, gamma=gamma
    )

    na_start_string = "Starting regular training of the resnet."
    adv_start_string = "Starting adversarial training of the resnet."

    return evaluate_na_adv(
        na_res_cls, na_start_string, adv_res_cls, adv_start_string,
        n_inputs, n_outputs, n_epochs,
        trainloader, testloader,
        na_log_path, na_save_path, adv_log_path, adv_save_path
    )


def evaluate_nn(
        n_hidden1: int,
        gamma: float,
        n_inputs: int,
        n_outputs: int,
        epsilon: float,
        n_epochs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        na_log_path: str,
        na_save_path: str,
        adv_log_path: str,
        adv_save_path: str,
    ) -> torch.Tensor:
    """
    Create and evaluate a one-hidden layer neural network.

    Inputs:
        n_hidden1: int
            Number of hidden nodes.

        gamma: float
            Initialization parameter controlling weight magnitudes.

        n_inputs: int
            Number of input dimensions

        n_outputs: int
            Number of outputs

        epsilon: float
            Adversarial noise magnitude.

        n_epochs: int
            Number of epochs.

        trainloader: DataLoader
            DataLoader for training data.

        testloader: DataLoader
            DataLoader for test data.

        na_log_path: str
            Log path for non-adversarial model.

        na_save_path: str
            Save path for the non-adversarial model.

        adv_log_path: str
            Log path for the adversarial results.

        adv_save_path: str
            Save path for the adversarially-trained model.

    Outputs:
        results: torch.Tensor
            Table containing results.
            The first row contains non-adversarial network results.
            The second row contains results for the adversarial network.
    """

    na_nn_cls = AdvOneLayer(
        n_inputs, n_outputs, n_hidden1, epsilon=epsilon, gamma=gamma
    )
    adv_nn_cls = AdvOneLayer(
        n_inputs, n_outputs, n_hidden1, epsilon=epsilon, gamma=gamma
    )

    na_start_string = "Starting regular training of the neural network."
    adv_start_string = "Starting adversarial training of the resnet."

    return evaluate_na_adv(
        na_nn_cls, na_start_string, adv_nn_cls, adv_start_string,
        n_inputs, n_outputs, n_epochs,
        trainloader, testloader,
        na_log_path, na_save_path, adv_log_path, adv_save_path
    )


def evaluate_all(
        n_hidden1: int,
        gamma: float,
        n_inputs: int,
        n_outputs: int,
        epsilon: float,
        n_epochs: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        na_lin_log_path: str,
        na_lin_save_path: str,
        adv_lin_log_path: str,
        adv_lin_save_path: str,
        na_res_log_path: str,
        na_res_save_path: str,
        adv_res_log_path: str,
        adv_res_save_path: str,
        na_nn_log_path: str,
        na_nn_save_path: str,
        adv_nn_log_path: str,
        adv_nn_save_path: str,
        results_path: str,
    ) -> torch.Tensor:
    """
    Create and evaluate a linear network, resnet, and one-layer network.

    Inputs:
        n_hidden1: int
            Number of hidden layers nodes in the hidden layer.

        gamma: float
            Inititalization weights parameter.

        n_inputs: int
            Number of input dimensions

        n_outputs: int
            Number of output dimensions

        epsilon: float
            Adversarial noise magnitude.

        n_epochs: int
            Number of epochs to train.

        trainloader: DataLoader
            Training data loader.

        testloader: DataLoader
            Test data loader.


        na_lin_log_path: str
            Non-adversarial linear model results path.

        na_lin_save_path: str
            Non-adversarial linear model save path.

        adv_lin_log_path: str
            Adversarial linear model log path.

        adv_lin_save_path: str
            Adversarial linear model save path.

        na_res_log_path: str
            Non-adversarial resnet results log path.

        na_res_save_path: str
            Non-adversarial resnet model save path.

        adv_res_log_path: str
            Adverarial resnet model results log path.

        adv_res_save_path: str
            Adversarial resnet model save path.

        na_nn_log_path: str
            Non-adversarial neural network log path.

        na_nn_save_path: str
            Non-adversarial neural network model save path.

        adv_nn_log_path: str
            Adversarial neural network log path.

        adv_nn_save_path: str
            Adversarial neural network model save path.

        results_path: str
            Save path for the accuracy results.

    Outputs:
        results: torch.Tensor
            Tensor containing results; first index is for model types.
            First is for linear, second is for the one-layer network,
            and the third is for the resnet.
    """

    results = torch.zeros((3, 2, 3))

    train_begin_string = "Beginning new training," \
        + "n_hidden1 = {0}, epsilon = {1}."
    print(train_begin_string.format(n_hidden1, np.around(epsilon, 2)))

    # Linear network
    results[0, :, :] = evaluate_linear(
        n_inputs, n_outputs, epsilon, n_epochs,
        trainloader, testloader,
        na_lin_log_path, na_lin_save_path, adv_lin_log_path, adv_lin_save_path
    )
    print(results)

    # One-layer nn
    results[1, :, :] = evaluate_nn(
        n_hidden1, gamma, n_inputs, n_outputs, epsilon, n_epochs,
        trainloader, testloader,
        na_nn_log_path, na_nn_save_path, adv_nn_log_path, adv_nn_save_path
    )
    print(results)

    # Resnet
    results[2, :, :] = evaluate_resnet(
        n_hidden1, gamma, n_inputs, n_outputs, epsilon, n_epochs,
        trainloader, testloader,
        na_res_log_path, na_res_save_path, adv_res_log_path, adv_res_save_path
    )
    print(results)

    torch.save(results, results_path)

    return results
