import torch

from experiment_mask_DLBM.v001.dlbm import (
    MaskedRelationDLBM, OrderedChangePointDensity, policy_from_scene_logits,
    representation_loss,
)


def make_taps(batch=2, layers=12, patches=16, channels=16):
    generator = torch.Generator().manual_seed(42)
    teacher = [
        torch.randn(batch, patches, channels, generator=generator)
        for _ in range(layers)
    ]
    student = [
        value + 0.2 * torch.randn(
            batch, patches, channels, generator=generator,
        )
        for value in teacher
    ]
    return teacher, student


def test_forward_backward_and_suffix_no_target_leak():
    model = MaskedRelationDLBM(
        feature_dim=16, token_dim=8, state_dim=8, heads=2, patch_count=16,
    )
    teacher, student = make_taps()
    output = model(teacher, student)
    assert output['distance'].shape == (2, 12, 16)
    assert output['prediction'].shape == (2, 11, 16)
    assert output['transition_token'].shape == (2, 12, 8)
    assert output['transition_set'].shape == (2, 12, 17, 8)
    assert output['transition_reconstruction'].shape == (2, 12, 16)
    loss, parts = representation_loss(output)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(float(value.abs().sum()) > 0 for value in gradients)

    # Prediction of L5 may use only L6..L11.  Changing the masked L5 observation must not alter it.
    model.eval()
    with torch.no_grad():
        original = model(teacher, student)['prediction'][:, 5]
        changed_teacher = list(teacher)
        changed_student = list(student)
        changed_teacher[5] = changed_teacher[5] + 20.0
        changed_student[5] = changed_student[5] - 10.0
        changed = model(changed_teacher, changed_student)['prediction'][:, 5]
    torch.testing.assert_close(original, changed, rtol=0, atol=0)


def test_ordered_density_recovers_known_suffix_change_point():
    density = OrderedChangePointDensity(token_dim=4, num_layers=12, min_scale=0.01)
    with torch.no_grad():
        density.means[0].fill_(-2.0)
        density.means[1].fill_(2.0)
        density.raw_scales.fill_(-4.0)
    tokens = torch.empty(32, 12, 4)
    tokens[:, :4] = -2.0
    tokens[:, 4:] = 2.0
    for values in (tokens, tokens.unsqueeze(2).expand(-1, -1, 17, -1)):
        logits = density.boundary_logits(values).mean(dim=0)
        policy = policy_from_scene_logits(logits)
        assert policy['boundary'] == 4
        assert policy['selected_taps'] == list(range(4, 12))
        assert 0.0 <= policy['normalized_entropy'] <= 1.0
    values = tokens.unsqueeze(2).expand(-1, -1, 17, -1)
    moments = {
        'global_mean': values[:, :, 0].mean(dim=0),
        'global_second': values[:, :, 0].square().mean(dim=0),
        'local_mean': values[:, :, 1:].mean(dim=(0, 2)),
        'local_second': values[:, :, 1:].square().mean(dim=(0, 2)),
    }
    _, moment_logits = density.scene_marginal_nll_from_moments(moments)
    assert policy_from_scene_logits(moment_logits)['boundary'] == 4
