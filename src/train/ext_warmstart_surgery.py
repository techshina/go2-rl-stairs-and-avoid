"""Build a bit-exact warm-start checkpoint for ObstacleExt from the stairs specialist (StairsV2).

Actor: the ObstacleExt input is laid out ``[proprio x5 (225) | near scan x5 (935) | far scan (422)]``
and the first 1,160 columns match the specialist's layout exactly (the near
block reproduces the stock 17x11 grid in the stock cell order, see
``extgrid_mdp.ext_height_scan_near``), so ``mlp.0.weight`` is a prefix copy
with zero columns for the far block.

Critic: the ObstacleExt critic sees the FULL extended grid (609 cells in extended
order) after the 60 proprio/privileged dims; the specialist's 187 scan columns
scatter into the extended order via ``extgrid_mdp.NEAR_IDX``; all other
extended cells get zero columns.

Both mappings are verified numerically below (must be exact to float precision).

Usage:
    isaaclab.sh -p src/train/ext_warmstart_surgery.py \
        --spec logs/rsl_rl/unitree_go2_velocity_stairsv2/<run>/model_9999.pt \
        --out logs/rsl_rl/unitree_go2_velocity_obstacleext/warmstart/model_0.pt
"""

import argparse
import os

import torch

# near-subgrid index map, inlined from extgrid_mdp (importing the package would
# pull the whole Isaac Lab task tree, which this pure-torch script doesn't need)
EXT_NX, NEAR_NX, NEAR_NY = 29, 17, 11
NEAR_IDX = tuple((iy + 5) * EXT_NX + ix for iy in range(NEAR_NY) for ix in range(NEAR_NX))

parser = argparse.ArgumentParser()
parser.add_argument("--spec", required=True, help="Stairs-specialist checkpoint (source weights).")
parser.add_argument("--out", required=True, help="Output checkpoint path (model_0.pt inside a new run dir).")
args = parser.parse_args()

N_PROPRIO_A, N_NEAR_HIST, N_FAR = 225, 935, 422  # actor blocks
N_PRIV_C, N_SCAN_C, N_EXT_C = 60, 187, 609  # critic blocks (spec scan / ext scan)

spec = torch.load(args.spec, weights_only=False, map_location="cpu")
new = {"iter": 0, "infos": spec.get("infos")}

# ---- actor: prefix copy + zero far block ----
a = {k: v.clone() for k, v in spec["actor_state_dict"].items()}
w = a["mlp.0.weight"]
assert w.shape == (512, N_PROPRIO_A + N_NEAR_HIST), w.shape
w_new = torch.zeros(512, N_PROPRIO_A + N_NEAR_HIST + N_FAR)
w_new[:, : w.shape[1]] = w
a["mlp.0.weight"] = w_new
new["actor_state_dict"] = a
print(f"[SURGERY] actor mlp.0: {tuple(w.shape)} -> {tuple(w_new.shape)} (far block zero)")

# ---- critic: scatter spec scan columns into the extended-grid order ----
c = {k: v.clone() for k, v in spec["critic_state_dict"].items()}
wc = c["mlp.0.weight"]
assert wc.shape == (512, N_PRIV_C + N_SCAN_C), wc.shape
wc_new = torch.zeros(512, N_PRIV_C + N_EXT_C)
wc_new[:, :N_PRIV_C] = wc[:, :N_PRIV_C]
near_idx = torch.tensor(NEAR_IDX, dtype=torch.long)
wc_new[:, N_PRIV_C + near_idx] = wc[:, N_PRIV_C:]
c["mlp.0.weight"] = wc_new
new["critic_state_dict"] = c
print(f"[SURGERY] critic mlp.0: {tuple(wc.shape)} -> {tuple(wc_new.shape)} (scan scattered via NEAR_IDX)")

# fresh optimizer: hyperparameters from the specialist's groups, moments empty
new["optimizer_state_dict"] = {"state": {}, "param_groups": spec["optimizer_state_dict"]["param_groups"]}


def forward(sd, x):
    for i in (0, 2, 4):
        x = torch.nn.functional.elu(x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
    return x @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]


torch.manual_seed(0)
# actor equivalence: shared prefix identical, far block arbitrary
xa = torch.randn(16, N_PROPRIO_A + N_NEAR_HIST)
xa_full = torch.cat([xa, 5.0 * torch.randn(16, N_FAR)], dim=1)
da = (forward(spec["actor_state_dict"], xa) - forward(a, xa_full)).abs().max()
# critic equivalence: near cells carry the spec scan values, all other ext cells arbitrary
xc = torch.randn(16, N_PRIV_C + N_SCAN_C)
xc_full = torch.zeros(16, N_PRIV_C + N_EXT_C)
xc_full[:, :N_PRIV_C] = xc[:, :N_PRIV_C]
xc_full[:, N_PRIV_C:] = 5.0 * torch.randn(16, N_EXT_C)
xc_full[:, N_PRIV_C + near_idx] = xc[:, N_PRIV_C:]
dc = (forward(spec["critic_state_dict"], xc) - forward(c, xc_full)).abs().max()
print(f"[SURGERY] equivalence: actor max|diff| = {da:.2e}, critic max|diff| = {dc:.2e}")
assert da < 1e-5 and dc < 1e-5, "surgery does not reproduce the specialist"

os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save(new, args.out)
print(f"[SURGERY] wrote {args.out}")
