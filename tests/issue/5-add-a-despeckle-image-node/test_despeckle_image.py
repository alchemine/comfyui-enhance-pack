import pytest
import torch


def flat(value=0.2, size=64):
    return torch.full((1, size, size, 3), value)


def with_specks(positions, size=64, brightness=0.9, halo=True):
    """A flat image with a 2x2 bright speck, and a slightly dark 1px halo, at each
    position. The halo stays within `threshold` of the background, as a real one
    does; a darker ring would be protected as an outline."""
    image = flat(size=size)
    for y, x in positions:
        if halo:
            image[0, y - 1 : y + 3, x - 1 : x + 3] = 0.15
        image[0, y : y + 2, x : x + 2] = brightness
    return image


def test_node_id_is_registered(mappings):
    assert "DespeckleImage" in mappings


def test_flat_image_passes_through(pack):
    image = flat()
    out, mask = pack("image").DespeckleImage.execute(image)

    assert torch.allclose(out, image, atol=1e-6) and mask.sum() == 0


def test_output_comes_back_on_the_input_device(pack):
    image = flat()
    out, mask = pack("image").DespeckleImage.execute(image)

    assert out.device == image.device and mask.device == image.device


def test_isolated_speck_is_removed_with_its_halo(pack):
    image = with_specks([(30, 30)])
    out, mask = pack("image").DespeckleImage.execute(image)

    assert mask[0, 30, 30] > 0.5
    assert torch.allclose(out[0, 24:38, 24:38], flat()[0, 24:38, 24:38], atol=0.02)


def test_speck_below_threshold_is_kept(pack):
    image = with_specks([(30, 30)], brightness=0.3, halo=False)
    out, mask = pack("image").DespeckleImage.execute(image, threshold=0.15)

    assert mask.sum() == 0 and torch.equal(out, image)


def test_line_is_kept(pack):
    image = flat()
    image[0, 30, 8:56] = 0.9
    out, mask = pack("image").DespeckleImage.execute(image)

    assert mask.sum() == 0 and torch.equal(out, image)


def test_dense_specks_are_kept(pack):
    image = with_specks([(y, x) for y in range(20, 44, 4) for x in range(20, 44, 4)])
    out, mask = pack("image").DespeckleImage.execute(image)

    assert mask.sum() == 0 and torch.equal(out, image)


def test_large_speck_is_kept(pack):
    image = flat()
    image[0, 28:38, 28:38] = 0.9
    out, mask = pack("image").DespeckleImage.execute(image, speck_size=7)

    assert mask.sum() == 0 and torch.equal(out, image)


def test_alpha_channel_is_kept(pack):
    image = with_specks([(30, 30)])
    rgba = torch.cat([image, torch.full(image.shape[:-1] + (1,), 0.25)], dim=-1)
    out, _ = pack("image").DespeckleImage.execute(rgba)

    assert out.shape[-1] == 4 and torch.equal(out[..., 3], rgba[..., 3])


def test_mask_has_image_shape(pack):
    image = with_specks([(30, 30)])
    _, mask = pack("image").DespeckleImage.execute(image)

    assert mask.shape == (1, 64, 64)


@pytest.mark.parametrize("brightness", [0.5, 0.9])
def test_speck_brightness_does_not_matter_above_threshold(pack, brightness):
    image = with_specks([(30, 30)], brightness=brightness)
    _, mask = pack("image").DespeckleImage.execute(image)

    assert mask[0, 30, 30] > 0.5


def test_faint_speck_on_black_is_removed(pack):
    image = flat(0.02)
    image[0, 30:32, 30:32] = 0.08
    out, mask = pack("image").DespeckleImage.execute(image, threshold=0.15)

    assert mask[0, 30, 30] > 0.5 and out[0, 30, 30, 0] < 0.03


def test_speck_next_to_a_bright_line_is_removed(pack):
    image = with_specks([(30, 30)])
    image[0, 8:56, 40] = 0.9
    out, mask = pack("image").DespeckleImage.execute(image)

    assert mask[0, 30, 30] > 0.5 and torch.equal(out[0, 8:56, 40], image[0, 8:56, 40])


def test_speck_next_to_a_thick_curved_edge_is_removed(pack):
    image = flat()
    for i in range(48):
        image[0, 8 + i, 8 + i // 2 : 11 + i // 2] = 0.9
    image[0, 30:32, 44:46] = 0.9
    out, mask = pack("image").DespeckleImage.execute(image)

    assert mask[0, 30, 44] > 0.5 and torch.equal(out[0, 20, 14:17], image[0, 20, 14:17])


def test_dominant_speck_in_texture_is_removed(pack):
    image = flat(0.05)
    for y in range(10, 54, 4):
        image[0, y, 10:54:3] = 0.3
    image[0, 30:32, 30:32] = 0.95
    out, mask = pack("image").DespeckleImage.execute(image)

    assert out[0, 30, 30, 0] < 0.7 and torch.equal(
        out[0, 14, 10:54], image[0, 14, 10:54]
    )


def test_speck_touching_a_faint_edge_on_black_is_removed(pack):
    image = flat(0.0)
    image[0, 8:56, 30] = 0.05
    image[0, 30:32, 31:33] = 0.9
    out, mask = pack("image").DespeckleImage.execute(image, threshold=0.08)

    assert mask[0, 30, 31] > 0.5 and torch.equal(out[0, 8:20, 30], image[0, 8:20, 30])


def test_dominant_speck_next_to_a_bright_area_is_kept(pack):
    image = flat(0.05)
    for y in range(10, 54, 4):
        image[0, y, 10:54:3] = 0.3
    image[0, 24:40, 40:54] = 0.4
    image[0, 30:32, 30:32] = 0.95
    out, mask = pack("image").DespeckleImage.execute(image)

    assert mask[0, 30, 30] == 0 and torch.equal(out, image)


def test_speck_barely_brighter_than_a_nearby_area_is_kept(pack):
    image = flat(0.3)
    image[0, 20:44, 30:60] = 0.6
    image[0, 25:27, 26:28] = 0.5
    image[0, 50:52, 10:12] = 0.9
    out, mask = pack("image").DespeckleImage.execute(image)

    assert mask[0, 25, 26] == 0 and mask[0, 50, 10] > 0.5
    assert torch.equal(out[0, 20:30, 20:30], image[0, 20:30, 20:30])


def test_highlight_at_a_bright_edge_is_kept(pack):
    image = flat(0.5)
    image[0, 20:44, 32:64] = 0.3
    image[0, 20:44, 30:32] = 0.75
    image[0, 25:27, 34:36] = 0.6
    out, mask = pack("image").DespeckleImage.execute(image)

    assert mask.sum() == 0 and torch.equal(out, image)
