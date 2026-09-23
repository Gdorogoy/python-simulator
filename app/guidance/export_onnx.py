"""Exports a saved ActorCritic checkpoint to ONNX so its graph can be
inspected in a viewer like Netron (https://netron.app).

A bare state_dict doesn't carry hidden/num_hidden_layers/dropout/obs_dim/
action_dim, so infer_architecture reads them straight from tensor shapes
(every `shared.*` Linear has `hidden` as out_features except the first,
whose in_features is obs_dim -- see ActorCritic.__init__) instead of
hardcoding each checkpoint's original training config. dropout doesn't
appear in the state_dict at all (no learned params), so it's irrelevant to
the exported inference graph regardless of what it was set to during
training.

Usage:
    python -m app.guidance.export_onnx <checkpoint.pt> [output.onnx]
"""
import os
import sys

import torch

from app.guidance.train import ActorCritic


def infer_architecture(state_dict: dict):
    """Returns (obs_dim, hidden, num_hidden_layers, action_dim)."""
    linear_idxs = sorted({
        int(k.split(".")[1]) for k in state_dict
        if k.startswith("shared.") and k.split(".")[2] == "weight"
    })
    first_w = state_dict[f"shared.{linear_idxs[0]}.weight"]
    obs_dim = first_w.shape[1]
    hidden = first_w.shape[0]
    num_hidden_layers = len(linear_idxs)
    action_dim = state_dict["actor_mean.weight"].shape[0]
    return obs_dim, hidden, num_hidden_layers, action_dim


def build_model_from_state_dict(state_dict: dict) -> ActorCritic:
    obs_dim, hidden, num_hidden_layers, action_dim = infer_architecture(state_dict)
    action_low = state_dict["action_low"]
    action_high = state_dict["action_high"]

    model = ActorCritic(obs_dim, action_dim, action_low, action_high,
                         hidden=hidden, num_hidden_layers=num_hidden_layers, dropout=0.0)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx_model(model: ActorCritic, obs_dim: int, output_path: str):
    """Exports an already-built, already-loaded model. Use this from a
    training loop that already has the model in memory -- no checkpoint
    round-trip needed."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    was_training = model.training
    model.eval()
    dummy_obs = torch.zeros(1, obs_dim, device=next(model.parameters()).device)
    torch.onnx.export(
        model, dummy_obs, output_path,
        input_names=["observation"], output_names=["action_mean", "action_std", "state_value"],
        dynamic_axes={"observation": {0: "batch"}, "action_mean": {0: "batch"},
                       "action_std": {0: "batch"}, "state_value": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # the dynamo exporter (torch>=2.x default) prints a noisy
                        # "[torch.onnx] ... done checkmark" progress log per export;
                        # the legacy TorchScript-based exporter does the same job quietly.
    )
    model.train(was_training)
    return output_path


def export_onnx_checkpoint(checkpoint_path: str, output_path: str = None) -> str:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_state_dict(state_dict)
    obs_dim = model.shared[0].in_features

    if output_path is None:
        output_path = os.path.splitext(checkpoint_path)[0] + ".onnx"

    export_onnx_model(model, obs_dim, output_path)
    print(f"[export_onnx] {checkpoint_path} -> {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    export_onnx_checkpoint(ckpt, out)
