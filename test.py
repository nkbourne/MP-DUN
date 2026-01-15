import argparse, os, glob
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch
from sources.util import instantiate_from_config
from utils import read_img, make_batch, psnr
from skimage.metrics import structural_similarity as ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=str,
        default="10",
        nargs="?",
        help="samping rate",
    )
    parser.add_argument(
        "--testset",
        type=str,
        default='Set5',
        nargs="?",
        help="dataset",
    )
    opt = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    setdir = "datasets/"+opt.testset+"/"
    images = glob.glob(os.path.join(setdir, "*.bmp"))
    print(f"Found {len(images)} inputs.",' in ', opt.testset)

    configpath = "models/S2/"+str(opt.sr)+"/config.yaml"
    ckptpath = "models/S2/"+str(opt.sr)+"/model.ckpt"
    config = OmegaConf.load(configpath)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckptpath, map_location="cuda")["state_dict"],strict=False)
    model.cuda()
    model.eval()

    outdir = "results/MP-DUN/"+opt.testset+"/" + str(opt.sr) + "/"
    os.makedirs(outdir, exist_ok=True)
    p_total = 0
    ssim_total = 0
    with torch.no_grad():
        for image in zip(images):
            outpath = os.path.join(outdir, os.path.split(image[0])[1])
            outpath = outpath.replace(".bmp",".png")
            outpath = outpath.replace(".jpg",".png")
            outpath = outpath.replace(".tif",".png")
            img = read_img(image[0])
            batch, h, w = make_batch(img, block_size = 32, channels=1, device=device)

            recon = model.implement(batch)
            recon_x = recon[:, :, 0:h, 0:w]

            ori_image = torch.clamp((img+1.0)*127.5, min=0.0, max=255.0)
            ori_image = ori_image.squeeze().cpu().numpy()
            predicted_image = torch.clamp((recon_x+1.0)*127.5, min=0.0, max=255.0)
            predicted_image = predicted_image.squeeze().cpu().numpy()
            Image.fromarray(predicted_image.astype(np.uint8)).save(outpath)
                
            p = psnr(ori_image,predicted_image)
            ss = ssim(predicted_image, ori_image, data_range=255)
            p_total = p_total + p
            ssim_total = ssim_total + ss
    print(opt.sr, ": PSNR: {:5.2f},SSIM: {:5.4f}".format( p_total/len(images), ssim_total/len(images)))
