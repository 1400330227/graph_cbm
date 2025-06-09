from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor

from graph_cbm.generalized_rcnn import GeneralizedRCNN
from graph_cbm.modeling.graph import Graph


class CBM(nn.Module):
    def __init__(
            self,
            transform: nn.Module,
            generalized_rcnn: GeneralizedRCNN,
            graph: Graph,
    ):
        super(CBM, self).__init__()
        self.generalized_rcnn = generalized_rcnn
        self.graph = graph

    def forward(
            self,
            images: List[Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None
    ):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])

    def postprocess(self, images: List[Tensor]) -> List[Tensor]:
        processed_images = []
        return processed_images

    def preprocess(self, images):
        print(images)



def build_cbm(cfg):
