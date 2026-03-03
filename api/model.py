import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int,
        mf_dim: int,
        mlp_user_dim: int,
        mlp_item_dim: int,
        genre_dim: int,
        mlp_linear_specs: List[Tuple[int,int]],
        mlp_input_dim: int,
    ):
        super().__init__()

        self.user_embedding_mf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_mf = nn.Embedding(num_items, mf_dim)

        self.user_embedding_mlp = nn.Embedding(num_users, mlp_user_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_item_dim)

        # checkpoint key typo must match exactly
        self.genre_embeddig = nn.Embedding(num_genres, genre_dim)

        # Build mlp_layers indices exactly as checkpoint (0 and 3 are Linear)
        max_idx = max(i for i, _ in mlp_linear_specs) if mlp_linear_specs else -1
        spec_map = {i: out for i, out in mlp_linear_specs}

        layers: List[nn.Module] = []
        cur_in = mlp_input_dim  # <-- FORCE to 527 based on ckpt
        for idx in range(max_idx + 1):
            if idx in spec_map:
                out_dim = spec_map[idx]
                layers.append(nn.Linear(cur_in, out_dim))
                cur_in = out_dim
            else:
                layers.append(nn.Identity())  # placeholder for dropout/relu/etc

        self.mlp_layers = nn.ModuleList(layers)
        self._mlp_out_dim = cur_in

        self.affine_output = nn.Linear(mf_dim + self._mlp_out_dim, 1)

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                x = torch.relu(layer(x))  # reasonable default
            else:
                x = layer(x)
        return x

    @torch.inference_mode()
    def predict_proba(self, user_id: int, item_id: int, genre_id: int) -> float:
        device = next(self.parameters()).device
        u = torch.tensor([user_id], dtype=torch.long, device=device)
        i = torch.tensor([item_id], dtype=torch.long, device=device)
        g = torch.tensor([genre_id], dtype=torch.long, device=device)

        # MF
        mf_vec = self.user_embedding_mf(u) * self.item_embedding_mf(i)

        # MLP input = [user256, item256, genre14] + [1.0]  => 527
        u_mlp = self.user_embedding_mlp(u)
        i_mlp = self.item_embedding_mlp(i)
        g_emb = self.genre_embeddig(g)
        ones = torch.ones((1, 1), dtype=u_mlp.dtype, device=device)
        mlp_in = torch.cat([u_mlp, i_mlp, g_emb, ones], dim=-1)

        mlp_vec = self.forward_mlp(mlp_in)

        x = torch.cat([mf_vec, mlp_vec], dim=-1)
        logit = self.affine_output(x).squeeze(-1)
        return float(torch.sigmoid(logit).item())

def build_model_from_state_dict(state: Dict[str, torch.Tensor]) -> NeuMF:
    # infer sizes from ckpt
    num_users = state["user_embedding_mf.weight"].shape[0]
    num_items = state["item_embedding_mf.weight"].shape[0]
    num_genres = state["genre_embeddig.weight"].shape[0]

    mf_dim = state["user_embedding_mf.weight"].shape[1]
    mlp_user_dim = state["user_embedding_mlp.weight"].shape[1]
    mlp_item_dim = state["item_embedding_mlp.weight"].shape[1]
    genre_dim = state["genre_embeddig.weight"].shape[1]

    # IMPORTANT: force mlp input dim from first linear layer weight
    mlp_input_dim = int(state["mlp_layers.0.weight"].shape[1])  # 527

    specs = []
    for k, v in state.items():
        if k.startswith("mlp_layers.") and k.endswith(".weight"):
            idx = int(k.split(".")[1])
            out_dim = int(v.shape[0])
            specs.append((idx, out_dim))
    specs.sort(key=lambda x: x[0])

    return NeuMF(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        mf_dim=mf_dim,
        mlp_user_dim=mlp_user_dim,
        mlp_item_dim=mlp_item_dim,
        genre_dim=genre_dim,
        mlp_linear_specs=specs,
        mlp_input_dim=mlp_input_dim,
    )
