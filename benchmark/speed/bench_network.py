import sys

sys.path.append("/home/sebastien/workspace/FaultyMemory")  # FIXME quick and dirty
import FaultyMemory as FyM
import torch
from utils import profile
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

resnet18 = models.resnet18(pretrained=True).to(device)
dummy_tensor = torch.randn([32, 3, 32, 32]).to(device)
representation = FyM.SlowFixedPointRepresentation()


def inference_parameters():
    handler = FyM.Handler(resnet18)
    handler.add_net_parameters(representation)
    handler(dummy_tensor)


_ = profile(inference_parameters, __file__, device)
