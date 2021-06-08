import numpy as np
import argparse
from os import path
import torch
import ssl
from glob import glob
from natsort import natsorted
ssl._create_default_https_context = ssl._create_unverified_context
import cv2


from utils.metrics import compute_fvd
from utils.general import get_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str,
                        required=True,
                        help="Source directory where the data is stored.")
    parser.add_argument("--gpu",type=int, required=True, help="The target device.")
    parser.add_argument("-v","--visualize",default=False,action="store_true")

    args = parser.parse_args()


    if not path.isdir(args.source):
        raise NotADirectoryError(f'The specified, data-holding directory {args.source} is not existing...')

    file = path.basename(__file__)
    logger = get_logger(file)

    logger.info("Read in data...")

    real_samples_list = natsorted(glob(path.join(args.source, "real_samples_*.npy")))
    fake_samples_list = natsorted(glob(path.join(args.source, "fake_samples_*.npy")))





    if len(real_samples_list) == 0:
        fake_samples_list = [path.join(args.source, "fake_samples.npy")]
        real_samples_list = [path.join(args.source, "real_samples.npy")]




    for i,(real_samples, fake_samples) in enumerate(zip(real_samples_list,fake_samples_list)):
        try:
            length = int(real_samples.split("/")[-1].split(".")[0].split("_")[2])
            context = int(real_samples.split("/")[-1].split(".npy")[0].split("_")[-1])
            logger.info(f"processing samples of length {length} with {context} context frames.")
        except:
            logger.info(f"Processing standard samples")

        real_samples = np.load(real_samples)
        fake_samples = np.load(fake_samples)

        if args.visualize:
            vis_real = real_samples[0,0]
            vis_fake = fake_samples[0,0]
            # visualize
            writer = cv2.VideoWriter(
                path.join(args.source, "test_vid_fake.mp4"),
                cv2.VideoWriter_fourcc(*"MP4V"),
                5,
                (vis_fake.shape[2], vis_fake.shape[1]),
            )

            # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

            for frame in vis_fake:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            writer.release()

            writer = cv2.VideoWriter(
                path.join(args.source, "test_vid_real.mp4"),
                cv2.VideoWriter_fourcc(*"MP4V"),
                5,
                (vis_real.shape[2], vis_real.shape[1]),
            )

            # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

            for frame in vis_real:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            writer.release()

        real_samples = real_samples[:62] if real_samples.shape[0] > 62 else real_samples
        fake_samples = fake_samples[:62] if fake_samples.shape[0] > 62 else fake_samples

        logger.info(f'Number of samples: {len(fake_samples)}')

        target_device = args.gpu

        real_samples = list(real_samples)
        fake_samples = list(fake_samples)
        real_samples = [torch.from_numpy(r) for r in real_samples]
        fake_samples = [torch.from_numpy(r) for r in fake_samples]



        fvd_val = compute_fvd(real_samples,fake_samples,target_device,logger)





