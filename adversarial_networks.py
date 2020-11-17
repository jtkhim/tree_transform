"""

Adversarial linear networks, one hidden layer neural networks, and
residual networks.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdvLinearNet(nn.Module):
    """
    Adversarial linear network.
    """
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            epsilon: float = 0.,
            q: int = 1
        ):
        """
        Initialize a linear network.

        Inputs:
            n_inputs: int
                Dimension of input.

            n_outputs: int
                Dimension of output.

            epsilon: float
                Adversarial noise magnitude.

            q: int
                Adversarial norm.
        """

        super(AdvLinearNet, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.epsilon = torch.tensor(epsilon)
        self.q = q
        self.beta = nn.Parameter(
            torch.rand((n_inputs, n_outputs), requires_grad=True) \
            / np.sqrt(n_outputs)
        )
        self.bias = nn.Parameter(torch.zeros(n_outputs, requires_grad=True))


    def adv_norm(self) -> torch.Tensor:
        """
        Calculate the adversarial norm, i.e., the perturbation size.

        Outputs:
            pert: torch.Tensor
                Size of adversarial perturbation.
        """

        return torch.norm(self.beta, dim=0, p=self.q)


    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            adv: bool,
        ) -> torch.Tensor:
        """
        Apply the linear network.

        Inputs:
            x: torch.Tensor
                Predictor tensor.

            y: torch.Tensor
                Target tensor.

            adv: bool
                Adversarial perturbation or not.

        Output:
            y_hat: torch.Tensor
                Predicted value of y.
        """

        base_output = torch.matmul(x, self.beta) + self.bias
        if adv:
            return base_output - self.epsilon * y * self.adv_norm()
        return base_output


    def get_epsilon(self) -> float:
        """
        Get the adversarial perturbation magnitude.

        Outputs:
            epsilon: float
                Magnitude of adversarial perturbation.
        """

        return self.epsilon


    def extra_repr(self) -> str:
        """
        Extra representation for the adversarial linear predictor.

        Output:
            info: str
                String providing number of inputs and outputs.
        """

        return 'n_inputs={0}, n_outputs={1}'.format(
            self.n_inputs, self.n_outputs
        )


class AdvOneLayer(nn.Module):
    """
    One hidden layer tree-transformed neural network.
    """

    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            n_hidden1: int,
            epsilon: float = 0.0,
            q: int = 1,
            gamma: float = 1.0
        ):
        """
        Initialize a one hidden layer neural network.

        Inputs:
            n_inputs: int
                Number of inputs.

            n_outputs: int
                Number of outputs.

            n_hidden1: int
                Number of hidden nodes.

            epsilon: float
                Adversarial perturbation size

            q: int
                Adversarial perturbation norm.

            gamma: float
                Normalizing factor.
        """

        super(AdvOneLayer, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden1 = n_hidden1
        self.epsilon = torch.tensor(epsilon)
        self.q = q

        # Initialize weights and biases
        self.W1 = nn.Parameter(
            -torch.rand((n_hidden1, n_inputs), requires_grad=True) \
            / (gamma * n_hidden1)
        )
        self.W2 = nn.Parameter(
            torch.rand((n_outputs, n_hidden1), requires_grad=True) \
            / (gamma * n_outputs)
        )
        self.bias1 = nn.Parameter(torch.zeros(n_hidden1, requires_grad=True))
        self.bias2 = nn.Parameter(torch.zeros(n_outputs, requires_grad=True))

        # Additional terms
        self.pert = None
        self.output = None


    def adv_norm_W1(self) -> torch.Tensor:
        """
        Compute the adversarial norm the first layer weights.

        Outputs:
            W1_norm: torch.Tensor
                Norm of first layer of weights.
        """

        return torch.norm(self.W1, p=self.q, dim=1)


    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor = None,
            adv: bool = False
        ) -> torch.Tensor:
        """
        Evaluate the network on data.

        Inputs:
            x: torch.Tensor
                Predictor tensor.

            y: torch.Tensor
                Target tensor. Unnecessary if not adversarial.

            adv: bool
                Whether or not to use an adversarial perturbation.

        Outputs:
            nn_output: torch.Tensor
                The predicted values of y.
        """

        if adv:
            batch_size = x.shape[0]
            W1_norm = self.adv_norm_W1()
            output = torch.zeros((batch_size, self.n_outputs, self.n_hidden1))
            pert_tensor = torch.zeros(
                (batch_size, self.n_outputs, self.n_hidden1)
            )

            for i in range(self.n_outputs):
                for j in range(self.n_hidden1):

                    adv_pert = -self.epsilon * y.float()[:, i] \
                        * self.W2[i, j].sign() * W1_norm[j]
                    hidden1 = F.relu(
                        torch.t(torch.matmul(self.W1[j, :], torch.t(x))) \
                        + self.bias1[j] + adv_pert.float()
                    )

                    summand = self.W2[i, j] * self.n_hidden1
                    output[:, i, j] = summand
                    pert_tensor[:, i, j] = adv_pert

            self.pert = pert_tensor
            self.output = output

            return torch.sum(output, dim=2) + self.bias2

        # If not adversarial
        hidden1 = torch.t(
            F.relu(torch.t(torch.matmul(self.W1, torch.t(x))) + self.bias1)
        )

        return torch.t(torch.matmul(self.W2, hidden1)) + self.bias2


    def get_epsilon(self) -> float:
        """
        Get adversarial perturbation magnitude.

        Outputs:
            epsilon: float
                Input perturbation magnitude.
        """

        return self.epsilon


    def extra_repr(self):
        """
        Extra representation for the network.

        Outputs:
            output: str
                String containing number of inputs, number of outputs,
                number of hidden layers, perturbation size, and
                perturbation norm.
        """

        output = 'n_inputs={}, n_outputs={}, n_hidden1={}, epsilon={}, p={}'
        return output.format(
            self.n_inputs, self.n_outputs, self.n_hidden1, self.epsilon, self.p
        )


class AdvResNet(nn.Module):
    """
    Adversarial resnet, i.e., a one hidden layer neural network and
    a linear component.
    """

    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            n_hidden1: int,
            epsilon: float = 0.0,
            p: int = 1,
            gamma: float = 1.0,
        ):
        """
        Initialize an adversarial resnet.

        Inputs:
            n_inputs: int
                Number of inputs.

            n_outputs: int
                Number of outputs.

            n_hidden1: int
                Number of hidden nodes.

            epsilon: float
                Adversarial perturbation size

            q: int
                Adversarial perturbation norm.

            gamma: float
                Normalizing factor.
        """

        super(AdvResNet, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden1 = n_hidden1
        self.epsilon = torch.tensor(epsilon)
        self.p = p
        self.one_layer = AdvOneLayer(
            n_inputs, n_outputs, n_hidden1, epsilon, p, gamma
        )
        self.linear = AdvLinearNet(n_inputs, n_outputs, epsilon, p)


    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor = None,
            adv: bool = False
        ) -> torch.Tensor:
        """
        Forward evalution of network.

        Inputs:
            x: torch.Tensor
                Predictor tensor.

            y: torch.Tensor
                Target tensor. Unnecessary if not adversarial.

            adv: bool
                Whether or not to use an adversarial perturbation.

        Outputs:
            nn_output: torch.Tensor
                The predicted values of y.
        """

        return self.linear(x, y, adv) + self.one_layer(x, y, adv)


    def get_epsilon(self) -> float:
        """
        Get the adversarial perturbation magnitude.

        Outputs:
            epsilon: float
                Magnitude of adversarial perturbation.
        """

        return self.epsilon
