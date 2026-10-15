import pytest
import torch
import torch.nn.functional as F


def image(height, width, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, height, width, 3, generator=g) * 0.3
    x[torch.rand(1, height, width, generator=g) > 0.995] = 1.0
    return x


@pytest.mark.parametrize("speck_size", [31, 21, 5])
def test_fill_convolves_the_mask_on_the_depthwise_path(pack, monkeypatch, speck_size):
    # CUDA runs conv2d as a depthwise kernel only when groups == channels > 1.
    # A single-channel conv2d goes through im2col instead and allocates
    # kernel_area * 4 bytes per pixel: over 10 GiB for 1024x1024 at 65x65.
    # Only the 5x5 feather of the mask may stay single-channel.
    mod = pack("image")
    calls = []
    conv2d_ = F.conv2d

    def conv2d(x, weight, *args, **kwargs):
        calls.append((x.shape[1], kwargs.get("groups", 1), weight.shape[-1]))
        return conv2d_(x, weight, *args, **kwargs)

    monkeypatch.setattr(mod.F, "conv2d", conv2d)
    mod.DespeckleImage.execute(image(64, 48), speck_size=speck_size)

    assert calls
    assert all(c == groups > 1 for c, groups, k in calls if k > 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fill_peak_memory_grows_with_the_image_not_the_kernel(pack):
    node = pack("image").DespeckleImage
    x = image(256, 256).cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    node.execute(x, speck_size=31)
    peak = torch.cuda.max_memory_allocated() - base

    assert peak < 200 * x.numel() * 4


def test_fill_matches_the_per_channel_formula(pack):
    # The filled image is the kernel-weighted mean of the kept pixels around
    # each speck, channel by channel.
    mod = pack("image")
    x = image(64, 48).movedim(-1, 1)
    mask = torch.zeros(1, 1, 64, 48)
    mask[:, :, 20:23, 10:13] = 1.0
    filled, feathered = mod._fill(x, mask, 0.08, 5, 3)

    grown = mod._dilate(mask, 7)
    closed = mod._erode(mod._dilate(x, 5), 5)
    dark = ((closed - x).amax(dim=1, keepdim=True) > 0.08).to(x.dtype)
    grown = grown * (1.0 - dark)
    kernel = mod._gaussian_kernel(6, 3.6)
    keep = 1.0 - grown
    conv = lambda t: mod._conv2d_per_channel(t.movedim(1, -1), kernel).movedim(-1, 1)
    expected = conv(x * keep) / conv(keep).clamp_min(1e-3)
    assert torch.allclose(
        filled, x * (1.0 - feathered) + expected * feathered, atol=1e-6
    )
