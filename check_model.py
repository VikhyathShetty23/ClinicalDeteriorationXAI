import torch

state = torch.load(
    "models/improved_lstm.pth",
    map_location="cpu",
    weights_only=True
)

if "state_dict" in state:
    state = state["state_dict"]

for key, value in state.items():
    print(key, value.shape)