import os, torch, torchvision, random, string
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

CURRENT_DIR = Path(__file__).parent.parent
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
NOTEBOOKS_DIR = os.path.join(CURRENT_DIR, "notebooks")
LOGS_DIR = os.path.join(CURRENT_DIR, "runs")
CKPT_DIR = os.path.join(CURRENT_DIR, "ckpt")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Device:", device)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    return device


def filter_str(str_, voc_type):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all':   string.digits + string.ascii_letters + string.punctuation
    }
    if voc_type == 'lower':
        str_ = str_.lower()
    for char in str_:
        if char not in alpha_dict[voc_type]:
            str_ = str_.replace(char, '')
    return str_


def tripple_display(image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, epoch, num_vis=5):
    """
    Display and save triplet comparison images during validation.
    
    Args:
        image_in: Low-resolution input images (batch) [B, C, H, W]
        image_out: Super-resolution output images (batch) [B, C, H, W]
        image_target: High-resolution target images (batch) [B, C, H, W]
        pred_str_lr: Predicted text from LR images (list of strings)
        pred_str_sr: Predicted text from SR images (list of strings)
        label_strs: Ground truth labels (list of strings)
        epoch: Current epoch number
        num_vis: Number of images to visualize (default: 5)
    """

    for i in range(min(image_in.shape[0], num_vis)):
        # Get RGB channels only (in case there are more channels)
        tensor_in = image_in[i][:3, :, :]
        
        # Resize LR input to match HR size for fair comparison
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (image_target.shape[-2], image_target.shape[-1]), 
                interpolation=Image.BICUBIC
            ),
            transforms.ToTensor()
        ])
        
        tensor_in = transform(tensor_in.cpu())
        tensor_out = image_out[i][:3, :, :].cpu()
        tensor_target = image_target[i][:3, :, :].cpu()
        
        # Stack images vertically (LR | SR | HR)
        images = [tensor_in, tensor_out, tensor_target]
        vis_im = torch.stack(images)
        vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
        
        # Create output directory structure
        out_root = os.path.join(CURRENT_DIR, 'demo', f'epoch_{epoch:03d}')
        os.makedirs(out_root, exist_ok=True)
        
        # Create filename with predictions and label
        # Format: LR-prediction_SR-prediction_GT-label_idx.png
        im_name = f"LR-{pred_str_lr[i]}_SR-{pred_str_sr[i]}_GT-{label_strs[i]}_{i:03d}.png"
        # Remove invalid characters for filename
        im_name = im_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # Save the image
        out_path = os.path.join(out_root, im_name)
        torchvision.utils.save_image(vis_im, out_path, padding=0)
        
    # print(f"Saved {min(image_in.shape[0], num_vis)} visualization images to {out_root}")