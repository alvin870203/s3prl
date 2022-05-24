import torch
# from torchinfo import summary
import s3prl.hub as hub

model = getattr(hub, "mockingjay")()  # build the Mockingjay model with pre-trained weights
# print(model)

wavs = [torch.randn(160000, dtype=torch.float) for _ in range(16)]

with torch.no_grad():
    reps = model(wavs)["hidden_states"]
    print(len(reps))
