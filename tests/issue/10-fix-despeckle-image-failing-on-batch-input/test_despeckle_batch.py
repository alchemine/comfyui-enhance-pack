import pytest
import torch


def batch(n, height, width, channels=3, seed=0):
    """Dark noise with scattered white specks, a different pattern per image."""
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, height, width, channels, generator=g) * 0.3
    images[torch.rand(n, height, width, generator=g) > 0.995] = 1.0
    return images


@pytest.mark.parametrize(
    "n, height, width, channels, speck_size",
    [
        (2, 64, 48, 3, 5),
        (4, 152, 104, 3, 5),
        (3, 100, 77, 4, 3),
        (5, 33, 65, 3, 7),
        (2, 128, 128, 3, 11),
    ],
)
def test_batch_matches_each_image_alone(pack, n, height, width, channels, speck_size):
    node = pack("image").DespeckleImage
    images = batch(n, height, width, channels)
    out, mask = node.execute(images, speck_size=speck_size)

    assert out.shape == images.shape and mask.shape == (n, height, width)
    for i in range(n):
        one_out, one_mask = node.execute(images[i : i + 1], speck_size=speck_size)
        assert torch.equal(out[i : i + 1], one_out)
        assert torch.equal(mask[i : i + 1], one_mask)


def test_blob_sizes_do_not_merge_across_images(pack):
    # The same blob at the same place in two images must be counted per image.
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, 2:4, 2:4] = 1.0
    sizes = pack("image")._blob_sizes(mask)

    assert torch.equal(sizes, mask * 4)


def test_blob_labels_stay_exact_past_float_precision(pack):
    # 17 images of 1216x832 hold more pixels than float32 counts exactly (2^24).
    # A lone pixel at the very end has an odd label past 2^24, which float32
    # rounds onto a neighbouring pixel.
    n, height, width = 17, 1216, 832
    mask = torch.zeros(n, 1, height, width)
    mask[:, :, -1, -1] = 1.0
    sizes = pack("image")._blob_sizes(mask)

    assert torch.equal(sizes, mask)
