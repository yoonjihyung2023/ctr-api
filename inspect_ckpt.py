import torch

path = "artifacts/model.pth"
sd = torch.load(path, map_location="cpu")
print("num_keys:", len(sd))
keys = [
    "user_embedding_mf.weight",
    "item_embedding_mf.weight",
    "user_embedding_mlp.weight",
    "item_embedding_mlp.weight",
    "genre_embeddig.weight",
    "mlp_layers.0.weight",
    "mlp_layers.0.bias",
    "mlp_layers.2.weight",
    "mlp_layers.2.bias",
    "mlp_layers.3.weight",
    "mlp_layers.3.bias",
    "affine_output.weight",
    "affine_output.bias",
]
for k in keys:
    if k in sd:
        print(k, tuple(sd[k].shape))
    else:
        print(k, "MISSING")
