import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms


class GradCAM(object):

    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.handlers = []
        self._register_hooks()
        self.feature = None
        self.gradient = None

    def _get_features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, grad_in, grad_out):
        self.gradient = grad_out[0]
        print("gradient shape:{}".format(grad_out[0].size()))

    def _register_hooks(self):
        for (name, module) in self.model.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handler in self.handlers:
            handler.remove()

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.model.zero_grad()
        output = self.model(inputs)
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        bz, nc, h, w = self.feature.shape
        size_upsample = (256, 256)

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature.cpu().data.numpy()

        feature_conv = feature.reshape((nc, h * w))
        weight_softmax = weight

        cam = weight_softmax.dot(feature_conv)
        cam = cam.reshape((h, w))
        cam = cam - np.min(cam)

        cam_img = cam / (np.max(cam) - np.min(cam))
        cam_img = np.uint8(255 * cam_img)
        cam_img = cv2.resize(cam_img, size_upsample)
        return cam_img


def get_net(net_name, weight_path=None):
    """
    It gets the model based on the net name.
    :param net_name:
    :param weight_path:
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name == 'resnet101':
        net = models.resnet101(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    elif net_name in ['mobilenet_v2']:
        net = models.mobilenet_v2(pretrained=pretrain)
    elif net_name in ['shufflenet_v2']:
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    print(net.eval())
    return net


def get_last_conv_name(net):
    """
    It gets the last convolution layer name of the net
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image_file):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    img_pil = Image.open(image_file)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))  # add the batch value
    return img_variable, img_tensor


def main(args):
    img = cv2.imread(args.image_path)
    height, width, _ = img.shape

    img_variable, img_tensor = prepare_input(args.image_path)

    net = get_net(args.network, args.weight_path)

    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(img_variable, args.class_id)
    heatmap = cv2.applyColorMap(cv2.resize(mask, (width, height)), cv2.COLORMAP_JET)
    cv2.imwrite('CAM_heatmap.jpg', heatmap)

    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', result)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Heatmap')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title('Result')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('grad_cam_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50')
    parser.add_argument('--image_path', type=str, default='')
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--layer-name', type=str, default=None, help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None, help='class id')

    arguments = parser.parse_args()
    main(arguments)
