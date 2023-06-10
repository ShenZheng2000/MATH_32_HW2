import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # `detach' the target from the graph
        # self.target = TODO
        pass

    def forward(self, input):
        # use l2 loss (i.e. mse loss)
        # self.loss = TODO
        pass


def gram_matrix(activations):
    # normalized_gram = TODO
    pass


"""Now the style loss module looks almost exactly like the content loss
module. The style distance is also computed using the mean square
error between $G_{XL}$ and $G_{SL}$.
"""


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # apply gram_matrix upon target_feature, then detach it.
        # self.target = TODO
        pass

    def forward(self, input):
        # apply gram_matrix upon input, then use l2 loss (i.e. mse loss)
        # self.loss = TODO
        pass
