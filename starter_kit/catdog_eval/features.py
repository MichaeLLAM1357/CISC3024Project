"""Inception-v3 pool3 features (2048-d), the standard feature space for FID, KID and
precision/recall on natural images (same network and weights as pytorch-fid and the
original TensorFlow FID code).

The weights (about 92 MB) are downloaded once by pytorch-fid into the torch hub cache
(~/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth). Without network
access, copy that file there by hand.

Images of any size are resized to 299 x 299 (bilinear) inside the network, so 32 x 32
images are upsampled; FID values are therefore not comparable with numbers computed at
other resolutions or with other resizing.
"""

from functools import lru_cache

import torch

FEATURE_DIM = 2048


@lru_cache(maxsize=None)
def load_inception(device: str = "cpu"):
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as err:
        raise SystemExit("the evaluator needs `pip install pytorch-fid`") from err
    block = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
    return InceptionV3([block], resize_input=True, normalize_input=True).to(device).eval()


@torch.no_grad()
def inception_features(images: torch.Tensor, device: str = "cpu",
                       batch_size: int = 200) -> torch.Tensor:
    """Features [N, 2048] (float64, on the CPU) of images [N, 3, H, W] in [0, 1]."""
    model = load_inception(str(device))
    out = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device, torch.float32)
        out.append(model(x)[0].flatten(1).double().cpu())
    return torch.cat(out) if out else torch.zeros(0, FEATURE_DIM, dtype=torch.float64)
