import cv2


def preprocess_image(img,swap_channels=False):
    """

    :param img: numpy array of shape (H,W,3)
    :param swap_channels: True, if channelorder is BGR
    :return:
    """
    if swap_channels:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # this seems to be possible as flownet2 outputs only images which can be divided by 64
    shape = img.shape
    img = img[:int(shape[0] / 64) * 64,:int(shape[1] / 64) * 64]

    return img