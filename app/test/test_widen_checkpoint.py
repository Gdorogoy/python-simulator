import torch

from app.guidance.train import ActorCritic, load_bc_checkpoint

LOW, HIGH = [-1.0] * 4, [1.0] * 4


def test_narrow_checkpoint_loads_into_wider_model(tmp_path):
    old = ActorCritic(23, 4, LOW, HIGH, hidden=16, num_hidden_layers=3)
    path = tmp_path / "old.pt"
    torch.save(old.state_dict(), path)

    wide = ActorCritic(23 + 7, 4, LOW, HIGH, hidden=16, num_hidden_layers=3)
    load_bc_checkpoint(wide, str(path))

    obs = torch.randn(5, 23)
    padded = torch.cat([obs, torch.randn(5, 7) * 100], dim=1)        # threat inputs are ignored at first
    assert torch.allclose(old(obs)[0], wide(padded)[0], atol=1e-6)
    assert wide.shared[0].weight[:, 23:].abs().sum() == 0


def test_wider_checkpoint_into_narrower_model_is_refused(tmp_path):
    big = ActorCritic(30, 4, LOW, HIGH, hidden=16, num_hidden_layers=3)
    torch.save(big.state_dict(), tmp_path / "big.pt")
    small = ActorCritic(23, 4, LOW, HIGH, hidden=16, num_hidden_layers=3)
    try:
        load_bc_checkpoint(small, str(tmp_path / "big.pt"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
