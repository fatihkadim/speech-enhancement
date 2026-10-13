"""src/models/unet.py testleri (M3 kabul kriterleri: boyutlar, skip, derinlik)."""
import pytest
import torch

from src.models.unet import UNet


def _params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("depth", [3, 4, 5])
@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("out_mode", ["mask", "direct"])
def test_output_shape_matches_input(depth: int, skip: bool, out_mode: str) -> None:
    model = UNet(base_ch=4, depth=depth, skip=skip, out_mode=out_mode).eval()
    x = torch.randn(2, 1, 256, 376)  # 3 sn'lik kesitin spektrogramı; 376, 2**depth'e bölünmez
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


@pytest.mark.parametrize("shape", [(1, 1, 256, 1), (1, 1, 257, 101), (3, 1, 64, 32)])
def test_odd_sizes_are_padded_and_cropped(shape: tuple[int, ...]) -> None:
    model = UNet(base_ch=4, depth=4).eval()
    with torch.no_grad():
        assert model(torch.randn(*shape)).shape == shape


def test_mask_in_unit_range() -> None:
    model = UNet(base_ch=4, depth=3, out_mode="mask").eval()
    with torch.no_grad():
        y = model(10 * torch.randn(2, 1, 64, 40))
    assert y.min() >= 0 and y.max() <= 1


def test_skip_off_keeps_architecture() -> None:
    """Skip kapatmak yalnızca bilgiyi kaldırır; parametre sayısı aynı kalmalı (adil E2 karşılaştırması)."""
    assert _params(UNet(base_ch=16, depth=4, skip=True)) == _params(UNet(base_ch=16, depth=4, skip=False))


def test_skip_changes_output() -> None:
    torch.manual_seed(0)
    a = UNet(base_ch=4, depth=3, skip=True).eval()
    b = UNet(base_ch=4, depth=3, skip=False).eval()
    b.load_state_dict(a.state_dict())  # aynı ağırlıklar
    x = torch.randn(1, 1, 64, 40)
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x))


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        UNet(out_mode="yanlis")
    with pytest.raises(ValueError):
        UNet(depth=0)
