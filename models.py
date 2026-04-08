from operator import le
import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


# np numpy
import numpy as np


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        # # pred = nn.Node()

        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"

        result = nn.as_scalar(self.run(x))

        if result >= 0:
            return 1
        return -1

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        all_trained = False
        while not all_trained:
            all_trained = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    all_trained = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        hidden_layers = 2
        hidden_layers_size = 200

        self.num_hidden_layers = hidden_layers

        self.weights = []
        self.biases = []

        layer_sizes = [1] + [hidden_layers_size] * hidden_layers + [1]

        for i in range(hidden_layers + 1):
            self.weights.append(nn.Parameter(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(nn.Parameter(1, layer_sizes[i + 1]))

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        hidden = x

        for i in range(self.num_hidden_layers):
            hidden = nn.Linear(hidden, self.weights[i])
            hidden = nn.AddBias(hidden, self.biases[i])
            hidden = nn.ReLU(hidden)

        output = nn.Linear(hidden, self.weights[self.num_hidden_layers])
        output = nn.AddBias(output, self.biases[self.num_hidden_layers])

        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"

        y_pred = self.run(x)
        return nn.SquareLoss(y, y_pred)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        all_params = self.weights + self.biases

        batch_size = dataset.x.shape[0] // 2
        worst_loss_scalar = float("inf")

        learning_rate = 0.1

        for x, y in dataset.iterate_forever(batch_size):
            # If any loss is less than 0.01, we can stop training
            if worst_loss_scalar < 0.01:
                break

            loss = self.get_loss(x, y)
            loss_scalar = nn.as_scalar(loss)

            if loss_scalar < worst_loss_scalar:
                worst_loss_scalar = loss_scalar

            grads = nn.gradients(loss, all_params)

            for param, grad in zip(all_params, grads):
                param.update(grad, -learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

        input_layer_size = 784
        output_layer_size = 10

        hidden_layers = 2
        hidden_layer_size = 128

        self.num_hidden_layers = hidden_layers
        self.weights = []
        self.biases = []

        layer_sizes = [input_layer_size] + [hidden_layer_size] * hidden_layers + [output_layer_size]

        for i in range(hidden_layers + 1):
            self.weights.append(nn.Parameter(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(nn.Parameter(1, layer_sizes[i + 1]))

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        hidden = x

        for i in range(self.num_hidden_layers):
            hidden = nn.Linear(hidden, self.weights[i])
            hidden = nn.AddBias(hidden, self.biases[i])
            hidden = nn.ReLU(hidden)

        output = nn.Linear(hidden, self.weights[self.num_hidden_layers])
        output = nn.AddBias(output, self.biases[self.num_hidden_layers])

        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        y_pred = self.run(x)
        return nn.SoftmaxLoss(y_pred, y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        all_params = self.weights + self.biases

        batch_size = 16
        learning_rate = 0.04

        for x, y in dataset.iterate_forever(batch_size):
            if dataset.get_validation_accuracy() > 0.974: # A bit higher then 97% to ensure we get above the 97% threshold
                break

            loss = self.get_loss(x, y)

            grads = nn.gradients(loss, all_params)

            for param, grad in zip(all_params, grads):
                param.update(grad, -learning_rate)
