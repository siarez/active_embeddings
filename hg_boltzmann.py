"""
2 layer Hourglass (hence `hg`) Restricted Boltzmann Machine
"""

import torch as t

class HGRBM(t.nn.Module):
    """
    A two layer fully connected RBM
    """
    def __init__(self, visible_size=32, hidden_size=8, bias=False):
        """
        The RBM has the hourglass structure visible_size x hidden_size x visible_size
        :param visible_size: Size of the ends (left & right) of the hourglass
        :param hidden_size: Size of the neck of the hourglass
        :param bias: Whether the model has biases
        """
        super().__init__()
        # todo: might need to turn these into `Parameter` at some point
        self.visible_size, self.hidden_size, self.bias = visible_size, hidden_size, bias
        k = t.sqrt(t.tensor(visible_size, dtype=t.float))  # init constant factor
        self.left_w = (2 * t.rand(visible_size, hidden_size) - 1.) / k
        self.right_w = (2 * t.rand(visible_size, hidden_size) - 1.) / k
        self.left_state = self.sample(t.rand(1, visible_size))
        self.right_state = self.sample(t.rand(1, visible_size))
        self.hidden_state = self.sample(t.rand(1, hidden_size))
        if bias:
            self.left_b = (2 * t.rand(1, visible_size)) / k
            self.right_b = (2 * t.rand(1, visible_size)) / k
            self.hidden_b = (2 * t.rand(1, hidden_size)) / k

    def inward(self, left=None, right=None):
        """
        Performs an inward pass from left and right, and updates the hidden state
        If left and right are none uses internal left&right states.
        :param left: Tensor to clamp left to. Shape should be (1, visible_size)
        :param right: Tensor to clamp right to. Shape should be (1, visible_size)
        :return: None
        """
        if left:
            self.left_state = left
        if right:
            self.right_state = right
        hidden_logit = self.left_state @ self.left_w + self.right_state @ self.right_w
        if self.bias:
            hidden_logit += self.hidden_b
        self.hidden_state = self.sample(t.sigmoid(hidden_logit))

    def outward(self):
        """
        Updates left and right states from the hidden state
        :return:
        """
        left_logit = self.hidden_state @ t.t(self.left_w)
        right_logit = self.hidden_state @ t.t(self.right_w)
        if self.bias:
            left_logit += self.left_b
            right_logit += self.right_b
        self.left_state = self.sample(t.sigmoid(left_logit))
        self.right_state = self.sample(t.sigmoid(right_logit))

    def energy(self):
        """
        Computes energy
        :return:
        """
        left_energy = self.left_state @ self.left_w @ t.t(self.hidden_state)
        right_energy = self.right_state @ self.right_w @ t.t(self.hidden_state)
        return left_energy + right_energy
    
    @staticmethod
    def sample(prob):
        return 2. * t.bernoulli(prob) - 1

