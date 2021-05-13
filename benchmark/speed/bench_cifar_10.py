import FaultyMemory as FyM
import torch
from benchmark.speed.utils import profile
from models.resnet import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

resnet = resnet18().to(device)
dataholder = FyM.utils.DataHolder.DataHolder()
opt_criterion = torch.nn.CrossEntropyLoss()

handler = FyM.Handler(resnet)
handler.add_net_parameters(FyM.SlowFixedPointRepresentation())

trainer = FyM.utils.Trainer.Trainer(handler, dataholder, opt_criterion, device, to_csv=True)


def inference_train_with_grads():
    trainer.loop(False, True)


def inference_train_without_grads():
    trainer.loop(False, False)


_ = profile(inference_train_with_grads, __file__, device)
_ = profile(inference_train_without_grads, __file__, device)
