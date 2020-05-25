"""
2 layer Hourglass (hence `hg`) Restricted Boltzmann Machine
"""

import torch as t


class HGRBM(t.nn.Module):
    """
    A two layer fully connected RBM
    """
    def __init__(self, visible_size=32, hidden_size=8, bias=False, temp=1.):
        """
        The RBM has the hourglass structure visible_size x hidden_size x visible_size
        :param visible_size: Size of the ends (left & right) of the hourglass
        :param hidden_size: Size of the neck of the hourglass
        :param bias: Whether the model has biases
        """
        super().__init__()
        # todo: might need to turn these into `Parameter` at some point
        self.visible_size, self.hidden_size, self.bias, self.temp = visible_size, hidden_size, bias, temp
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
        if left is not None:
            self.left_state = left
        if right is not None:
            self.right_state = right
        hidden_logit = self.left_state @ self.left_w + self.right_state @ self.right_w
        if self.bias:
            hidden_logit += self.hidden_b
        self.hidden_state = self.sample(t.sigmoid(hidden_logit / self.temp))

    def outward(self, just_logit=False):
        """
        Updates left and right states from the hidden state
        :return:
        """
        left_logit = self.hidden_state @ t.t(self.left_w)
        right_logit = self.hidden_state @ t.t(self.right_w)
        if self.bias:
            left_logit += self.left_b
            right_logit += self.right_b
        if just_logit:
            # this is for the case where they are chain together. In that case, the left logits from
            # the right hgrbm and right logits from the left hgrbm need to be sumed before taking a sample
            return left_logit, right_logit
        else:
            self.left_state = self.sample(t.sigmoid(left_logit / self.temp))
            self.right_state = self.sample(t.sigmoid(right_logit / self.temp))

    def energy(self):
        """
        Computes energy
        :return: energy per sample in the batch
        """
        # Todo:
        left_w_energy = ((self.left_state @ self.left_w) * self.hidden_state).sum(dim=-1, keepdim=True)
        right_w_energy = ((self.right_state @ self.right_w) * self.hidden_state).sum(dim=-1, keepdim=True)
        left_b_energy = self.left_state @ t.t(self.left_b)
        right_b_energy = self.right_state @ t.t(self.right_b)
        hidden_b_energy = self.hidden_state @ t.t(self.hidden_b)
        return -(left_w_energy + right_w_energy + left_b_energy + right_b_energy + hidden_b_energy)

    def train_cd(self, x_left, x_right, lr=0.001, k=2):
        """
        Trains the network for one step with contrastive divergence of `k` steps for negative sampling.
        :param x: Input. The first dimension is batch
        :param lr: learning rate
        :param k: number of steps for negative sampling
        :return: Returns weight update-size l1 norms; to be used as a measure of convergence
        """
        batch_size = t.tensor(x_left.shape[0], dtype=t.float)
        lr = lr / batch_size  # Normalizing learning rate for batch size
        self.inward(x_left, x_right)
        h_positive = self.hidden_state.clone()
        for _ in range(k):
            self.outward()
            self.inward()
        # updating weights
        grad_left_w = t.t(x_left) @ h_positive - t.t(self.left_state) @ self.hidden_state
        grad_right_w = t.t(x_right) @ h_positive - t.t(self.right_state) @ self.hidden_state
        self.left_w += grad_left_w * lr
        self.right_w += grad_right_w * lr
        # updating biases
        grad_left_b = (x_left - self.left_state).sum(dim=0)
        grad_right_b = (x_right - self.right_state).sum(dim=0)
        grad_hidden_b = (h_positive - self.hidden_state).sum(dim=0)
        self.left_b += grad_left_b * lr
        self.right_b += grad_right_b * lr
        self.hidden_b += grad_hidden_b * lr

        return t.norm(grad_left_w, p=1) * lr, t.norm(grad_right_w, p=1) * lr

    @staticmethod
    def sample(prob):
        return 2. * t.bernoulli(prob) - 1


