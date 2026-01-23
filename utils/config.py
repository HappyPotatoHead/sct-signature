import cv2 as cv

import torchvision.models as models # pyright: ignore[reportMissingTypeStubs]

from typing import Dict, Tuple, Any
from dataclasses import dataclass

@dataclass
class CLAHEConfig:
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)

@dataclass
class BlurConfig:
    kernel_size: Tuple[int, int] = (5, 5)
    
@dataclass 
class ThresholdConfig:
    method: int = cv.THRESH_BINARY + cv.THRESH_OTSU

BACKBONE_CONFIG: Dict[str, Dict[str, Any]] = {
    "efficientnet_v2_s": {"builder": models.efficientnet_v2_s, "out_channels": 1280},
    "efficientnet_v2_m": {"builder": models.efficientnet_v2_m, "out_channels": 1280},
    "efficientnet_v2_l": {"builder": models.efficientnet_v2_l, "out_channels": 1280},
}

HF_REPO_ID = "HairyPotato/signature-models"

MODELS: Dict[str, Dict[str, str | float]] = {
    "SCT+": {
        "ckpt": "25_auc_0.1051.pt",
        "threshold": 0.725,
        "description": "Best overall performance on CEDAR"
    },
    "Batch Semi Hard Mining": {
        "ckpt": "14_auc_0.0769.pt",
        "threshold": 0.777,
        "description": "Second best performance on CEDAR"
    },
    "Batch Hard Mining": {
        "ckpt": "5_auc_0.0693.pt",
        "threshold": 0.667,
        "description": "Worst performance on CEDAR"
    }
}