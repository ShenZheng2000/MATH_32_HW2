import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss
import argparse
import torch.nn.functional as F
import os


# desired depth layers to compute style/content losses :
layer_dict = {'1': 'conv_1', '2': 'conv_2', '3': 'conv_3', '4': 'conv_4', '5': 'conv_5'}


def get_base(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_model_and_losses(cnn, style_img, content_img, content_layers, style_layers):

    content_layers = [layer_dict[layer] for layer in content_layers]
    style_layers = [layer_dict[layer] for layer in style_layers]

    cnn = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []

    normalization = Normalization().to(device)
    model = nn.Sequential(normalization)

    i = 0

    for layer in cnn.children():

        if isinstance(layer, nn.Conv2d):
            name = f'conv_{i}'
            i += 1
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
            
    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""

def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1,
                     content_layers=None, style_layers=None):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses

    # get the optimizer

    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function
    # def closure():
    # here
    # which does the following:
    # clear the gradients
    # compute the loss and it's gradient
    # return the loss

    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step

    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img, content_layers, style_layers)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_image_optimizer(input_img)

    print('==============Optimizing==============>')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            if use_style:
                for sl in style_losses:
                    style_score += sl.loss
                style_score *= style_weight

            if use_content:
                for cl in content_losses:
                    content_score += cl.loss
                content_score *= content_weight

            loss = (style_score + content_score)
            loss.backward()

            run[0] += 1

            return loss

        optimizer.step(closure)

    # make sure to clamp once you are done
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def main(style_img_path, content_img_path, output_path, task, content_layers, style_layers, content_weight, style_weight):
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)

    # Get the dimensions and number of channels of the two images
    _, channels1, height1, width1 = content_img.size()
    _, channels2, height2, width2 = style_img.size()
    
    # convert from gray to color
    if channels2 == 1:
        style_img = style_img.repeat(1, 3, 1, 1)

    # if content has larger h or w, upscale the style image
    if width1 >= width2 or height1 >= height2:
        style_img = F.interpolate(style_img, size=(height1, width1), mode='bilinear', align_corners=True)
    
    # print("content_img shape is", content_img.shape)
    # print("style_img shape is", style_img.shape)
        
    # interative MPL
    # plt.ion()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    print("Performing Style Transfer from content image initialization")
    # input_img = content_img.clone()
    # output = transfer the style from the style_img to the content image
    input_img = content_img.clone()
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, 
                            style_weight=style_weight, content_weight=content_weight, 
                            content_layers=content_layers, style_layers=style_layers)

    plt.figure()
    imshow(output, title='Output Image from content')

    save_dir = f'{args.output_path}/{args.task}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.tight_layout()
    plt.savefig(f"{args.output_path}/{args.task}/"
            f"{get_base(args.content_img_path)}"
            f"_{get_base(args.style_img_path)}"
            f"_{args.content_weight}_{args.style_weight:.0e}.jpg",
            bbox_inches='tight')



def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # Input Image Path
    parser.add_argument('--style_img_path', type=str, default="./images/style/escher_sphere.jpeg")
    parser.add_argument('--content_img_path', type=str, default="./images/content/dancing.jpg")
    parser.add_argument('--output_path', type=str, default="./images/output")
    parser.add_argument('--task', type=str, default='style_transfer_content')
    parser.add_argument('--content_layers', nargs='+', type=str, default='4')
    parser.add_argument('--style_layers', nargs='+', type=str, default=['1','2','3','4','5'])
    parser.add_argument('--content_weight', type=int, default=1)
    parser.add_argument('--style_weight', type=float, default=1e5)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(**vars(args))
