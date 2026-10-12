import pytest
import torch


@pytest.fixture
def image():
    torch.manual_seed(0)
    return torch.rand(1, 16, 24, 3)


def test_node_id_is_registered(mappings):
    assert "AdjustImage" in mappings


def test_node_is_in_enhance_pack_category(mappings):
    assert mappings["AdjustImage"].CATEGORY == "EnhancePack/Image"


def test_defaults_pass_the_image_through(pack, image):
    (out,) = pack("image").AdjustImage.execute(image)

    assert torch.equal(out, image)


def test_brightness_scales_the_image(pack, image):
    (out,) = pack("image").AdjustImage.execute(image, brightness=0.5)

    assert torch.allclose(out, image * 0.5)


def test_saturation_zero_makes_grayscale(pack, image):
    (out,) = pack("image").AdjustImage.execute(image, saturation=0.0)

    assert torch.allclose(out[..., 0], out[..., 1]) and torch.allclose(
        out[..., 1], out[..., 2]
    )


def test_alpha_channel_is_kept(pack, image):
    rgba = torch.cat([image, torch.full(image.shape[:-1] + (1,), 0.25)], dim=-1)
    (out,) = pack("image").AdjustImage.execute(rgba, contrast=1.5, cas=0.5)

    assert out.shape[-1] == 4 and torch.equal(out[..., 3], rgba[..., 3])


def test_output_stays_in_range(pack, image):
    (out,) = pack("image").AdjustImage.execute(
        image, brightness=3.0, contrast=3.0, edge_enhance=1.0, local_contrast=2.0
    )

    assert out.min() >= 0.0 and out.max() <= 1.0


def test_scale_by_resizes(pack, image):
    (out,) = pack("image").AdjustImage.execute(image, scale_by=1.5)

    assert out.shape == (1, 24, 36, 3)
