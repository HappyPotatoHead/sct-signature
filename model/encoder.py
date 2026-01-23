from utils.config import BACKBONE_CONFIG

import sys
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model_weights # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

class FeatureExtractionModel(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        embedding_dim: int = 256,
        weights: Optional[str] = None, 
        dropout_rate: float  = 0.4,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._check_parameters(backbone_type, embedding_dim, dropout_rate)
        
        FREEZE_UP_TO: int = 4
        
        self.embedding_dim = embedding_dim
        self.backbone_type = backbone_type
        self.dropout_rate = dropout_rate  
        
        self.weights = weights
        self.weights_enum = None
        
        backbone_builder = BACKBONE_CONFIG[self.backbone_type]["builder"]
        backbone_out_channels = BACKBONE_CONFIG[self.backbone_type]["out_channels"]
        
        self._retrieve_weights(str(self.weights), backbone_builder)
        
        self.model = backbone_builder(weights = self.weights_enum)
        
        first_layer = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels= 1,
            out_channels=first_layer.out_channels,
            kernel_size=first_layer.kernel_size,
            stride=first_layer.stride,
            padding=first_layer.padding,
            bias=first_layer.bias is not None,
        )
        
        if self.weights is not None:
            with torch.no_grad():
                self.model.features[0][0].weight.data = (
                    first_layer.weight.data.mean(dim=1, keepdim=True)
                )
        else:
            nn.init.kaiming_normal_(
                self.model.features[0][0].weight, mode="fan_out", nonlinearity="relu"
            )
            if self.model.features[0][0].bias is not None:
                nn.init.constant_(self.model.features[0][0].bias, 0)

        self.backbone = self.model.features

        # Zero shot Learning       
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        for i, block in enumerate(self.model.features):
            if i <= FREEZE_UP_TO:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
            
        self.pool1 = nn.AdaptiveAvgPool2d(1) 
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(backbone_out_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, self.embedding_dim)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim) 
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        
        x = self.pool1(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        # x = self.relu(x)
        
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def _check_parameters(
        self,
        backbone_type: str,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError(
                f"Embedding dimension must be positive, got {embedding_dim}"
            )
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(
                f"Dropout rate must be between 0 and 1, got {dropout_rate}"
            )
        if backbone_type not in BACKBONE_CONFIG:
            raise ValueError(
                f"Unsupported backbone type: {backbone_type}. Choose from {list(BACKBONE_CONFIG.keys())}"
            )
    
    def _retrieve_weights(
        self, 
        weights: str, 
        backbone_builder: Any
    ) -> None:
        try:
            weights_enum_type = get_model_weights(backbone_builder)
            if hasattr(weights_enum_type, weights):
                self.weights_enum = getattr(weights_enum_type, weights)
            else:
                print(
                    f"Warning: Could not get weights type for backbone '{self.backbone_type}'. Using default random initialisation.",
                    file=sys.stderr,
                )
                weights = ""
        except AttributeError:
            print(
                f"Warning: Specified weights alias '{weights}' not found for {self.backbone_type}. Check available weights in torchvision documentation. Using default random initialisation.",
                file=sys.stderr,
            )
            weights = ""
        except Exception as e:
            print(
                f"Warning: An unexpected error occurred looking up weights '{weights}' for {self.backbone_type}: {e}. Using default random initialisation.",
                file=sys.stderr,
            )
            
            weights = ""
 