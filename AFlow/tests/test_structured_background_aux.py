import copy
import torch
from sammed3d.structured_background_aux import StructuredBackgroundLocator, conditional_rc_loss, export_plain_state
from sammed3d.t1ce_wt_capg3d import AutomaticPromptLocator
from scripts.structured_background_protocol import check_roles


def test_authorized_roles_exclude_all_protected_patients():
    fit, dev = check_roles()
    assert len(fit) == 684 and len(dev) == 180


def test_auxiliary_gradient_is_zero_inside_wt_and_nonzero_outside():
    logits = torch.zeros(1,1,2,2,2, requires_grad=True)
    wt = torch.zeros_like(logits); wt[...,0,0,0] = 1
    rc = torch.zeros_like(logits); rc[...,1,1,1] = 1
    loss = conditional_rc_loss(logits, rc, wt)
    loss.backward()
    assert logits.grad[...,0,0,0] == 0
    assert logits.grad[...,1,1,1] < 0
    assert torch.isfinite(logits.grad).all()


def test_all_wt_and_empty_rc_loss_is_finite():
    logits = torch.zeros(2,1,2,2,2,requires_grad=True)
    loss = conditional_rc_loss(logits, torch.zeros_like(logits), torch.ones_like(logits))
    assert float(loss) == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_export_preserves_only_primary_output_and_plain_parameter_count():
    torch.set_num_threads(1); torch.manual_seed(7)
    model = StructuredBackgroundLocator().eval()
    plain = AutomaticPromptLocator(center_aware=False).eval()
    plain.load_state_dict(export_plain_state(model), strict=True)
    image = torch.randn(1,1,16,16,16)
    with torch.no_grad():
        left = model(image)[0]; right = plain(image).segmentation_logits
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert sum(v.numel() for v in plain.parameters()) == 4700897


def test_zero_coefficient_reproduces_primary_only_gradients():
    a = torch.randn(1,2,2,2,2, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    wt = torch.zeros_like(a[:,:1]); rc = torch.ones_like(wt)
    (a[:,:1].square().mean() + 0 * conditional_rc_loss(a[:,1:],rc,wt)).backward()
    b[:,:1].square().mean().backward()
    torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_patient_cluster_bootstrap_preserves_repeated_scan_groups():
    from scripts.evaluate_structured_background_aux import bootstrap
    rows=[{'subject_id':'a','dice':.2},{'subject_id':'a','dice':.4},{'subject_id':'b','dice':.9}]
    result=bootstrap(rows,19)
    # With two clusters the extrema are both-a and both-b draws; a's scans stay together.
    torch.testing.assert_close(torch.tensor(result['subject_95ci']),torch.tensor([.3,.9]))
    torch.testing.assert_close(torch.tensor(result['scan_cluster_95ci']),torch.tensor([.3,.9]))
