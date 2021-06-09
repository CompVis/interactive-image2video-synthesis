import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_lightning.metrics import functional as PF
from tqdm import tqdm



class FIDInceptionModel(nn.Module):
    def __init__(self, normalize_range=True):
        super().__init__()
        self.v3 = inception_v3(pretrained=True,aux_logits=False)
        # self.v3.aux_logits = False


        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
                .unsqueeze(dim=0)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
                .unsqueeze(dim=0)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1),
        )

        self.resize = nn.Upsample(size=(299,299),mode="bilinear")
        self.normalize_range = normalize_range

    def forward(self, x):
        x = self.resize(x)
        if self.normalize_range:
            # normalize in between 0 and 1
            x = (x + 1.) / 2.
        else:
            x = x.to(torch.float) / 255.
        # normalize to demanded values
        x = (x - self.mean) / self.std

        # this reimpleents the respective layers of the inception model, see model definition
        for name, submodule in self.v3._modules.items():
            x = submodule(x)
            if name == "Mixed_7c":
                break
            elif name == "Conv2d_4a_3x3" or name == "Conv2d_2b_3x3":
                x = F.avg_pool2d(x, kernel_size=3, stride=2)

        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = torch.flatten(out, 1)

        return out

def metrcis_MAE(tensor1, tensor2, mean=True):
    value = torch.sum(torch.abs(tensor1 - tensor2))
    if mean:
        value /= tensor1.view((-1)).shape[0]
    return value.data.cpu().numpy()


def metrcis_MSE(tensor1, tensor2, mean=True):
    diff = tensor1 - tensor2
    value = torch.sum(diff * diff)
    if mean:
        value /= tensor1.view((-1)).shape[0]
    return value.data.cpu().numpy()


def metrcis_l1(tensor1, tensor2, mean=False):
    return metrcis_MAE(tensor1, tensor2, mean)


def metrcis_l2(tensor1, tensor2, mean=False):
    diff = tensor1 - tensor2
    value = torch.sum(diff * diff)
    value = torch.sqrt(value)
    if mean:
        value /= tensor1.view((-1)).shape[0]
    return value.data.cpu().numpy()


def metric_ssim(real, fake, reduce = True, return_per_frame=False):
    if real.dim() == 3:
        real = real[None,None]
        fake = fake[None,None]
    elif real.dim() == 4:
        real = real[None]
        fake = fake[None]
    # rescale to valid range
    real = ((real + 1.) / 2.).permute(0, 1, 3, 4, 2).cpu().numpy()
    fake = ((fake + 1.) / 2.).permute(0, 1, 3, 4, 2).cpu().numpy()


    ssim_batch = np.asarray([ssim(rimg, fimg, multichannel=True, data_range=1.0,
                                  gaussian_weights=True,use_sample_covariance=False, ) for rimg, fimg in zip(real.reshape(-1,*real.shape[2:]),
                                                                                                             fake.reshape(-1,*fake.shape[2:]))])

    if return_per_frame:
        ssim_per_frame = {}
        for i in range(real.shape[1]):
            real_test = real[:,i]
            fake_test = fake[:,i]
            ssim_per_frame[i] = np.asarray([ssim(real_test, fake_test,
                                 multichannel=True, data_range=1., gaussian_weights=True, use_sample_covariance=False)])
        # ssim_per_frame = {i:np.asarray([ssim(real[:,i], fake[:,i],
        #                          multichannel=True, data_range=1., gaussian_weights=True, use_sample_covariance=False)]) for i in range(real.shape[1])}

    if reduce:
        if return_per_frame:
            ssim_pf_reduced = {key: ssim_per_frame[key] for key in ssim_per_frame}
            return np.mean(ssim_batch), ssim_pf_reduced
        else:
            return np.mean(ssim_batch)
    if return_per_frame:
        return ssim_batch, ssim_per_frame
    else:
        return ssim_batch

def ssim_lightning(real, fake, return_per_frame=False, normalize_range=True):
    if real.dim() == 3:
        real = real[None, None]
        fake = fake[None, None]
    elif real.dim() == 4:
        real = real[None]
        fake = fake[None]

    if normalize_range:
        real = (real + 1.) /2.
        fake = (fake + 1.) / 2.

    ssim_batch = PF.ssim(fake.reshape(-1,*fake.shape[2:]),real.reshape(-1,*real.shape[2:])).cpu().numpy()

    if return_per_frame:
        ssim_per_frame = {i: PF.ssim(fake[:,i],real[:,i]).cpu().numpy() for i in range(real.shape[1])}

        return ssim_batch, ssim_per_frame

    return ssim_batch


def psnr_lightning(real, fake, return_per_frame=False, normalize_range=True):
    if real.dim() == 3:
        real = real[None, None]
        fake = fake[None, None]
    elif real.dim() == 4:
        real = real[None]
        fake = fake[None]
    if normalize_range:
        real = (real + 1.) / 2.
        fake = (fake + 1.) / 2.

    psnr_batch = PF.psnr(fake.reshape(-1, *fake.shape[2:]), real.reshape(-1, *real.shape[2:])).cpu().numpy()

    if return_per_frame:
        psnr_per_frame = {i: PF.psnr(fake[:, i].contiguous(), real[:, i].contiguous()).cpu().numpy() for i in range(real.shape[1])}

        return psnr_batch, psnr_per_frame

    return psnr_batch

def metric_psnr(im_true, im_test,reduce = True, return_per_frame=False):
    if im_true.dim() == 3:
        im_true, im_test = im_true[None,None], im_test[None,None]
    elif im_true.dim() == 4:
        im_true, im_test = im_true[None], im_test[None]
    # make channel last
    real = ((im_true + 1.) / 2.).permute(0, 1, 3, 4, 2).cpu().numpy()
    fake = ((im_test + 1.) / 2.).permute(0, 1, 3, 4, 2).cpu().numpy()

    psnr_batch = np.asarray([compare_psnr(r,f, data_range=1.) for r, f in zip(real.reshape(-1,*real.shape[2:]),fake.reshape(-1,*fake.shape[2:]))])

    if return_per_frame:
        psnr_per_frame = {i: np.asarray([compare_psnr(real[:,i], fake[:,i], data_range=1.)]) for i in range(real.shape[1])}


    if reduce:
        if return_per_frame:
            psnr_pf_reduced = {key: psnr_per_frame[key] for key in psnr_per_frame}
            return np.mean(psnr_batch), psnr_pf_reduced
        else:
            return np.mean(psnr_batch)
    if return_per_frame:
        return psnr_batch, psnr_per_frame
    else:
        return psnr_batch

def metric_lpips(real, fake, lpips_func, reduce=True, return_per_frame=False, normalize=False):
    if real.dim() == 3:
        real, fake = real[None,None], fake[None,None]
    elif real.dim() == 4:
        real, fake = real[None], fake[None]

    if normalize:
        if fake.max() > 1:
            fake = (fake.to(torch.float) / 127.5) - 1.
            real = (real.to(torch.float) / 127.5) -1.
        else:
            real = (real * 2.) - 1.
            fake = (fake * 2.) - 1.


    lpips_batch = lpips_func(real.reshape(-1,*real.shape[2:]),fake.reshape(-1,*fake.shape[2:])).squeeze().cpu().numpy()
    if return_per_frame:
        lpips_per_frame = {i: lpips_func(real[:,i],fake[:,i]).squeeze().cpu().numpy() for i in range(real.shape[1])}

    if reduce:
        if return_per_frame:
            lpips_pf_reduced = {key: lpips_per_frame[key].mean() for key in lpips_per_frame}
            return lpips_batch.mean(), lpips_pf_reduced
        else:
            return lpips_batch.mean()

    if return_per_frame:
        return lpips_batch, lpips_per_frame
    else:
        return lpips_batch


def mean_cov(features):
    mu = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mu,cov




def metric_fid(real_features, fake_features, eps=1e-6):
    # Taken and adapted from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py

    if not isinstance(real_features,np.ndarray):
        real_features = np.concatenate(real_features,axis=0)
        fake_features = np.concatenate(fake_features,axis=0)

    mu1, cov1 = mean_cov(real_features)
    mu2, cov2 = mean_cov(fake_features)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(cov1)
    sigma2 = np.atleast_2d(cov2)

    assert (
            mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
            sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"

        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def compute_fvd(real_videos,fake_videos, device,logger):
    import silence_tensorflow.auto
    import tensorflow.compat.v1 as tf
    from utils.frechet_video_distance import preprocess,Embedder,calculate_fvd
    # required for fvd computation


    # config = tf.ConfigProto()
    # config.gpu_options.visible_device_list = f"{device}"



    devs = tf.config.experimental.get_visible_devices("GPU")
    target_dev = [d for d in devs if d.name.endswith(str(device))][0]
    tf.config.experimental.set_visible_devices(target_dev, 'GPU')
    logger.info("Compute fvd score.")
    #dev = f"/gpu:{device}"
    logger.info(f"using device {device}")
    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            # construct graph
            sess = tf.Session()

            input_shape = real_videos[0].shape
            input_real = tf.placeholder(dtype=tf.uint8, shape=input_shape)
            input_fake = tf.placeholder(dtype=tf.uint8, shape=input_shape)

            real_pre = preprocess(input_real, (224, 224))

            emb_real = Embedder(real_pre)
            embed_real = emb_real.create_id3_embedding(real_pre)
            fake_pre = preprocess(input_fake, (224, 224))
            emb_fake = Embedder(fake_pre)
            embed_fake = emb_fake.create_id3_embedding(fake_pre)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            real, fake = [], []
            for rv, fv in tqdm(zip(real_videos, fake_videos)):
                # real_batch = ((rv + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy()
                # fake_batch = ((fv + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy()
                # real_batch = ((rv + 1.) * 127.5).cpu().numpy()
                # fake_batch = ((fv + 1.) * 127.5).cpu().numpy()
                feed_dict = {input_real: rv, input_fake: fv}
                r, f = sess.run([embed_fake, embed_real], feed_dict)
                real.append(r)
                fake.append(f)
            print('Compute FVD score')
            real = np.concatenate(real, axis=0)
            fake = np.concatenate(fake, axis=0)
            embed_real = tf.placeholder(dtype=tf.float32, shape=(real.shape[0], 400))
            embed_fake = tf.placeholder(dtype=tf.float32, shape=(real.shape[0], 400))
            result = calculate_fvd(embed_real, embed_fake)
            feed_dict = {embed_real: real, embed_fake: fake}
            fvd_val = sess.run(result, feed_dict)
            sess.close()


            logger.info(f"Results of fvd computation: fvd={fvd_val}")

    # for being sure
    return fvd_val




if __name__ == "__main__":
    z, o = torch.rand((1080, 720, 3)), torch.rand((1080, 720, 3))
    o[0, 0, 0], o[1, 0, 0] = 0, 1
    z[0, 0, 0], z[1, 0, 0] = 0, 1
