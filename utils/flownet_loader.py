import torch
from torch.nn import functional as F
from PIL import Image
from models.flownet2.models import *

from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

from utils.general import get_gpu_id_with_lowest_memory


class FlownetPipeline:
    def __init__(self):
        super(FlownetPipeline, self).__init__()

    def load_flownet(self, args, device):
        """

        :param args: args from argparser
        :return: The flownet pytorch model
        """
        # load model savefile
        save = torch.load(
            "/export/data2/ablattma/Datasets/plants/pretrained_models/FlowNet2_checkpoint.pth.tar")
        model = FlowNet2(args, batchNorm=False)

        untrained_statedict = model.state_dict()

        # load it into proper clean model
        model.load_state_dict(save["state_dict"])
        model.eval()
        return model.to(device)

    def preprocess_image(self, img, img2, channelOrder="RGB",spatial_size= None):
        """ This preprocesses the images for FlowNet input. Preserves the height and width order!

        :param channelOrder: RGB(A) or BGR
        :param img: The first image in form of (W x H x RGBA) or (H x W x RGBA)
        :param img2: The first image in form of (W x H x RGBA) or (H x W x RGBA)
        :return: The preprocessed input for the prediction (BGR x Img# x W x H) or (BGR x Img# x H x W)
        """
        # ToTensor transforms from (H x W x C) => (C x H x W)
        # also automatically casts into range [0, 1]
        if spatial_size is None:
            img, img2 = transforms.ToTensor()(img)[:3], transforms.ToTensor()(img2)[:3]
        else:
            ts = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=spatial_size,interpolation=Image.BILINEAR),transforms.ToTensor()])
            img, img2 = ts(img)[:3],ts(img2)[:3]
        if channelOrder == "RGB":
            img, img2 = img[[2, 1, 0]], img2[[2, 1, 0]]

        # Cast to proper shape (Batch x BGR x #Img x H x W)
        s = img.shape
        img, img2 = img[:, :int(s[1] / 64) * 64, :int(s[2] / 64) * 64], \
                    img2[:, :int(s[1] / 64) * 64,:int(s[2] / 64) * 64]
        stacked = torch.cat([img[:, None], img2[:, None]], dim=1)
        return stacked

    def predict(self, model, stacked, spatial_size=None):
        """

        :param stacked: The two input images. (Batch x BGR x Img# x H x W)
        :return: The flow result (2 x W x H)
        """
        # predict
        model.eval()
        prediction = model(stacked)
        out_size = float(prediction.shape[-1])
        if spatial_size is not None:
            prediction = F.interpolate(
                prediction.cpu(), size=(spatial_size,spatial_size), mode="bilinear"
            )
            # rescale to make it fit to new shape (not grave, if this is skipped as flow is normalized anyways later)
            prediction = prediction / (out_size / spatial_size)
        flow = prediction[0]
        return flow

    def show_results(self, prediction, with_ampl=False):
        """

        prediction (Tensor): The predicted flow (2 x W x H)
        :return: plots
        """

        zeros = torch.zeros((1, prediction.shape[1], prediction.shape[2]))
        if with_ampl:
            ampl = torch.sum(prediction * prediction, dim=0)
            ampl = ampl.squeeze()
        else:
            ampl = torch.cat([prediction, zeros], dim=0)
        ampl -= ampl.min()
        ampl /= ampl.max()

        # show image
        im = transforms.ToPILImage()(ampl)
        if with_ampl:
            plt.imshow(im, cmap='gray')
        else:
            plt.imshow(im)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description='Process some integers.')
    # always 1.0, because pytorch toTensor automatically converts into range [0.0, 1.0]
    parser.add_argument("--rgb_max", type=float, default=1.)
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    args = parser.parse_args()

    # load test images in BGR mode
    img, img2 = np.asarray(Image.open(f"/export/data/ablattma/Datasets/plants/processed/hoch_misc1/frame_0.png")), \
                np.asarray(Image.open(f"/export/data/ablattma/Datasets/plants/processed/hoch_misc1/frame_100.png"))

    # load Flownet
    pipeline = FlownetPipeline()
    flownet_device = get_gpu_id_with_lowest_memory()
    flownet = pipeline.load_flownet(args, flownet_device)

    # process to show flow
    stacked = pipeline.preprocess_image(img, img2).to(flownet_device)
    prediction = pipeline.predict(flownet, stacked[None]).cpu()
    pipeline.show_results(prediction)
    plt.show()