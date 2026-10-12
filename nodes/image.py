"""Nodes in EnhancePack/Image."""

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


#################################################################
# Base class
#################################################################
class BaseImage:
    """Base class for Image nodes."""


#################################################################
# Helpers
#################################################################
def _rgb_to_luminance(image: torch.Tensor) -> torch.Tensor:
    """Return per-pixel luminance with a trailing channel dim kept.

    image: (..., H, W, 3) in [0, 1]
    """
    weights = torch.tensor(
        [0.299, 0.587, 0.114], device=image.device, dtype=image.dtype
    )
    return (image[..., :3] * weights).sum(dim=-1, keepdim=True)


def _conv2d_per_channel(rgb: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a 2D kernel to each channel independently (reflect padding).

    rgb:    (B, H, W, C) in [0, 1]
    kernel: (kh, kw)
    """
    x = rgb.movedim(-1, 1)  # (B, C, H, W)
    c = x.shape[1]
    kh, kw = kernel.shape
    weight = kernel.to(device=x.device, dtype=x.dtype).expand(c, 1, kh, kw)
    padded = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
    # cuDNN autotuning (`--fast autotune`) spends over a second on the first
    # call per image size, far more than these small depthwise kernels save.
    with torch.backends.cudnn.flags(benchmark=False):
        out = F.conv2d(padded, weight, groups=c)
    return out.movedim(1, -1)


def _cas(rgb: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Contrast Adaptive Sharpening (AMD FidelityFX CAS).

    Adaptive unsharp: sharpens flat regions more and already-sharp edges less,
    so it avoids the crunchy over-sharpening of a fixed kernel.

    rgb:       (B, H, W, C) in [0, 1]
    sharpness: 0.0 = softest sharpen ... 1.0 = strongest
    """
    x = rgb.movedim(-1, 1)  # (B, C, H, W)
    p = F.pad(x, (1, 1, 1, 1), mode="reflect")
    up = p[..., :-2, 1:-1]
    down = p[..., 2:, 1:-1]
    left = p[..., 1:-1, :-2]
    right = p[..., 1:-1, 2:]
    center = x

    mn = torch.minimum(
        torch.minimum(torch.minimum(up, down), torch.minimum(left, right)), center
    )
    mx = torch.maximum(
        torch.maximum(torch.maximum(up, down), torch.maximum(left, right)), center
    )

    # Adaptive weight: less sharpening where local contrast is already high.
    d = torch.clamp(torch.minimum(mn, 1.0 - mx) / (mx + 1e-6), 0.0, 1.0)
    amp = torch.sqrt(d)
    # sharpness lerps the base weight between gentle (-0.125) and strong (-0.2).
    w = amp * (-0.125 - 0.075 * sharpness)

    out = (center + (up + down + left + right) * w) / (1.0 + 4.0 * w)
    return out.movedim(1, -1)


def _local_contrast(rgb: torch.Tensor, amount: float, radius: int = 15) -> torch.Tensor:
    """Large-radius unsharp mask ("clarity"): boosts mid-scale volume/depth."""
    kernel = torch.ones(radius, radius) / float(radius * radius)
    blurred = _conv2d_per_channel(rgb, kernel)
    return rgb + amount * (rgb - blurred)


def _bilateral(rgb: torch.Tensor, strength: float, radius: int = 2) -> torch.Tensor:
    """Edge-preserving denoise (bilateral filter approximation).

    Smooths flat noise/banding while keeping lines and edges crisp.
    strength: 0.0 = off ... 1.0 = strong smoothing.
    """
    x = rgb.movedim(-1, 1)  # (B, C, H, W)
    p = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    _, _, ph, pw = p.shape
    h, w = x.shape[-2:]

    sigma_space = radius
    # Stronger strength => wider color sigma => smooths across bigger differences.
    sigma_color = 0.05 + 0.35 * strength

    acc = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = p[
                ..., radius + dy : radius + dy + h, radius + dx : radius + dx + w
            ]
            spatial = torch.exp(
                torch.tensor(-(dy * dy + dx * dx) / (2.0 * sigma_space**2))
            )
            color = torch.exp(-((neighbor - x) ** 2) / (2.0 * sigma_color**2))
            weight = spatial * color
            acc += neighbor * weight
            wsum += weight
    smoothed = acc / (wsum + 1e-8)
    smoothed = smoothed.movedim(1, -1)
    return rgb * (1.0 - strength) + smoothed * strength


#################################################################
# Nodes
#################################################################
class AdjustImage(BaseImage):
    """Adjust brightness, contrast, saturation, gamma and a suite of sharpen/
    denoise filters (edge enhance, CAS, local contrast, bilateral), plus
    optional upscaling of an image.

    All adjustments run as torch ops on the device ComfyUI computes on, and
    the result comes back on the device the image arrived on. RGB channels only;
    an existing alpha channel (RGBA) is passed through untouched. Every knob
    defaults to a no-op so the node is a pass-through until a slider is moved.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "brightness": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "1.0 = no change, <1.0 = darker, >1.0 = brighter.",
                },
            ),
            "contrast": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "1.0 = no change, <1.0 = less contrast, >1.0 = more contrast.",
                },
            ),
            "saturation": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "1.0 = no change, 0.0 = grayscale, >1.0 = more vivid.",
                },
            ),
            "gamma": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "1.0 = no change, <1.0 = brighter mid-tones, >1.0 = darker mid-tones.",
                },
            ),
            "edge_enhance": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend factor for edge enhancement. 0.0 = off, 1.0 = full.",
                },
            ),
            "cas": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Contrast Adaptive Sharpening. 0.0 = off. Adaptive, avoids crunchy edges. Great for anime.",
                },
            ),
            "local_contrast": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Large-radius clarity. 0.0 = off. Adds mid-scale depth/volume.",
                },
            ),
            "denoise": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Edge-preserving denoise (bilateral). 0.0 = off. Removes flat noise/banding, keeps lines.",
                },
            ),
            "upscale_method": (
                UPSCALE_METHODS,
                {
                    "default": "lanczos",
                    "tooltip": "Resampling method used when scale_by != 1.0.",
                },
            ),
            "scale_by": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.05,
                    "tooltip": "Scale factor. 1.0 = no resize, 1.2 = 20% larger, <1.0 = smaller.",
                },
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "EnhancePack/Image"

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        gamma: float = 1.0,
        edge_enhance: float = 0.0,
        cas: float = 0.0,
        local_contrast: float = 0.0,
        denoise: float = 0.0,
        upscale_method: str = "lanczos",
        scale_by: float = 1.0,
    ) -> tuple[torch.Tensor]:
        device = image.device
        image = image.to(comfy.model_management.get_torch_device())
        has_alpha = image.shape[-1] == 4
        alpha = image[..., 3:] if has_alpha else None
        rgb = image[..., :3].clone()

        # Brightness
        if brightness != 1.0:
            rgb = rgb * brightness

        # Contrast (around mid-gray 0.5)
        if contrast != 1.0:
            rgb = (rgb - 0.5) * contrast + 0.5

        # Saturation (interpolate between luminance and color)
        if saturation != 1.0:
            lum = _rgb_to_luminance(rgb)
            rgb = lum + (rgb - lum) * saturation

        rgb = rgb.clamp(0.0, 1.0)

        # Gamma (needs non-negative input, so applied after clamp)
        if gamma != 1.0:
            rgb = rgb.pow(gamma)

        # Denoise first (edge-preserving), so later sharpening doesn't amplify noise.
        if denoise > 0.0:
            rgb = _bilateral(rgb, denoise)

        # Edge enhance (EDGE_ENHANCE_MORE kernel, blended by factor)
        if edge_enhance > 0.0:
            kernel = torch.tensor(
                [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]]
            )
            enhanced = _conv2d_per_channel(rgb, kernel)
            rgb = rgb * (1.0 - edge_enhance) + enhanced * edge_enhance

        # Contrast Adaptive Sharpening (adaptive, edge-safe)
        if cas > 0.0:
            rgb = _cas(rgb, cas)

        # Local contrast / clarity (large-radius unsharp)
        if local_contrast > 0.0:
            rgb = _local_contrast(rgb, local_contrast)

        rgb = rgb.clamp(0.0, 1.0)

        out = torch.cat([rgb, alpha], dim=-1) if has_alpha else rgb

        # Upscale / resize
        if scale_by != 1.0:
            _, h, w, _ = out.shape
            new_w = max(1, round(w * scale_by))
            new_h = max(1, round(h * scale_by))
            samples = out.movedim(-1, 1)  # (B, H, W, C) -> (B, C, H, W)
            samples = comfy.utils.common_upscale(
                samples, new_w, new_h, upscale_method, "disabled"
            )
            out = samples.movedim(1, -1).clamp(0.0, 1.0)

        return (out.to(device),)


#################################################################
# Despeckle
#################################################################
def _dilate(x: torch.Tensor, size: int) -> torch.Tensor:
    """Grayscale dilation with a square kernel. x: (B, C, H, W)"""
    pad = size // 2
    return F.max_pool2d(
        F.pad(x, (pad, pad, pad, pad), mode="replicate"), size, stride=1
    )


def _erode(x: torch.Tensor, size: int) -> torch.Tensor:
    return -_dilate(-x, size)


def _blob_sizes(mask: torch.Tensor, iters: int = 12) -> torch.Tensor:
    """Size of the 8-connected blob each set pixel belongs to. mask: (B, 1, H, W)
    of 0/1. Labels are the smallest flat index in the blob, spread by min
    pooling plus pointer jumping, so a blob a few hundred pixels long settles
    in a dozen rounds."""
    b, _, h, w = mask.shape
    n = b * h * w
    on = mask > 0
    idx = torch.arange(n, device=mask.device).view(b, 1, h, w)
    sentinel = torch.full_like(idx, n)
    lab = torch.where(on, idx, sentinel)
    for _ in range(iters):
        lab = torch.where(on, -F.max_pool2d(-lab.double(), 3, 1, 1).long(), sentinel)
        flat = torch.cat([lab.flatten(), sentinel.flatten()[:1]])
        lab = flat[flat[lab.flatten()]].view(b, 1, h, w)
    sizes = torch.bincount(lab.flatten(), minlength=n + 1).to(mask.dtype)
    sizes[n] = 0
    return sizes[lab.flatten()].view(b, 1, h, w)


def _ring_max(x: torch.Tensor, outer: int, inner: int) -> torch.Tensor:
    """Max of x over the square ring between `inner` and `outer` pixels away
    from each pixel: the patch minus the window a speck itself can fill."""
    _, _, h, w = x.shape
    p = F.pad(x, (outer, outer, outer, outer), value=0.0)
    band = outer - inner
    rows = F.max_pool2d(p, (band, 2 * outer + 1), stride=1)
    cols = F.max_pool2d(p, (2 * outer + 1, band), stride=1)
    return torch.maximum(
        torch.maximum(
            rows[..., :h, :w], rows[..., inner + 1 + outer :, :w][..., :h, :]
        ),
        torch.maximum(
            cols[..., :h, :w], cols[..., :h, inner + 1 + outer :][..., :, :w]
        ),
    )


def _gaussian_kernel(radius: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def _count(x: torch.Tensor, size: int) -> torch.Tensor:
    """Sum of x over the `size` square window around each pixel."""
    return F.avg_pool2d(
        x, size, stride=1, padding=size // 2, count_include_pad=False
    ) * (size * size)


# A patch (opened image mean) darker than this is near black: the threshold
# scales down with its brightness, never below SPECK_FLOOR, and the pop-out
# rule is skipped.
DARK_PATCH = 0.1
SPECK_FLOOR = 0.03
# A patch brighter than this is skin or hair: any bright pixel counts as crowd.
BRIGHT_PATCH = 0.2
# Off a near-black patch a speck must be this many times brighter than
# everything the opening left within `speck_size` of it.
POPOUT_RATIO = 1.5
# A candidate at least this fraction of the brightest top-hat within
# `speck_size` is traced as a blob among such candidates only.
STRONG_FRACTION = 0.25
# A speck on a patch with nothing brighter than BLACK_PATCH, with a top-hat
# above DOMINANT_TOPHAT and DOMINANT_RATIO times any other candidate in the
# patch, is removed however crowded the patch is.
BLACK_PATCH = 0.06
DOMINANT_TOPHAT = 0.3
DOMINANT_RATIO = 2.0
# A speck is isolated when every patch (4 x speck_size + 1) containing it holds
# at most MAX_PIXELS other speck pixels; its dark halo is filled too (grow).
MAX_PIXELS = 20


def _speck_mask(
    x: torch.Tensor, threshold: float, speck_size: int, patch_size: int, max_pixels: int
) -> torch.Tensor:
    """0/1 mask (B, 1, H, W) of the pixels that are specks. x: (B, C, H, W)"""
    opened = _dilate(_erode(x, speck_size), speck_size)
    background = opened.mean(dim=1, keepdim=True)
    local_mean = _count(background, patch_size) / (patch_size * patch_size)
    tophat = (x - opened).amax(dim=1, keepdim=True)

    # Candidates: brighter than the surroundings by `threshold`, less on a
    # near-black patch.
    limit = torch.clamp(threshold * local_mean / DARK_PATCH, SPECK_FLOOR, threshold)
    candidate = (tophat > limit).to(x.dtype)

    # Pop-out: fabric texture and the tip of a lit fold are only a little
    # brighter than the cloth they sit on; salt on a sky is several times
    # brighter. Such pixels still count as crowd below.
    around = F.max_pool2d(background, 2 * speck_size + 1, stride=1, padding=speck_size)
    popout = (local_mean < DARK_PATCH) | (
        x.mean(dim=1, keepdim=True) >= POPOUT_RATIO * (around + 0.02)
    )

    # Small: a blob that fits in a window twice `speck_size`. Anything longer
    # (a lit edge, a strand, a carving) is never removed. Strong candidates are
    # traced among themselves so a white dot touching a far fainter edge is
    # not taken for part of it; the faint ones are traced among all
    # candidates so the faint edge is still a line.
    peak = F.max_pool2d(tophat, 2 * speck_size + 1, stride=1, padding=speck_size)
    strong = candidate * (tophat >= STRONG_FRACTION * peak).to(x.dtype)
    small = torch.zeros_like(candidate)
    for group, traced in ((strong, strong), (candidate - strong, candidate)):
        near = _count(traced, 2 * speck_size + 1)
        small += group * (near + 0.5 >= _blob_sizes(traced)).to(x.dtype)

    # Isolated: every patch containing the pixel holds few other specks and
    # few candidates of any kind (hair, texture), so the end of a line counts
    # the whole line. On a bright patch (skin, hair) a lit edge counts as
    # crowd too, so a highlight at a jawline is not taken for a speck.
    own = _count(small, speck_size)
    crowd = _dilate(_count(small, patch_size), patch_size) - own
    total = _dilate(_count(candidate, patch_size), patch_size)
    isolated = (crowd <= max_pixels) & (total <= 4 * max_pixels)
    isolated &= (local_mean < BRIGHT_PATCH) | (total - own <= max_pixels)

    # Dominant: a white dot between the letters of a dark book spine. An eye
    # highlight sits on a dark pupil too, but the iris next to it is brighter
    # than BLACK_PATCH.
    window = (patch_size, 1, patch_size // 2)
    black = F.max_pool2d(background, *window) < BLACK_PATCH
    others = torch.maximum(
        _ring_max(tophat * candidate, patch_size // 2, speck_size),
        F.max_pool2d(tophat * candidate * (1.0 - small), *window),
    )
    dominant = black & (tophat > DOMINANT_TOPHAT) & (tophat >= DOMINANT_RATIO * others)

    return small * (popout & (isolated | dominant)).to(x.dtype)


def _fill(
    x: torch.Tensor, mask: torch.Tensor, threshold: float, speck_size: int, grow: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the masked pixels and their dark halo. Returns the filled image
    and the feathered mask, both (B, C|1, H, W)."""
    if grow > 0:
        mask = _dilate(mask, 2 * grow + 1)
    # Growing must not swallow a dark line next to the speck (a mouth, an
    # outline). A speck's faint halo stays below `threshold`.
    closed = _erode(_dilate(x, speck_size), speck_size)
    dark = ((closed - x).amax(dim=1, keepdim=True) > threshold).to(x.dtype)
    mask = mask * (1.0 - dark)

    # Normalized convolution from the pixels around the speck continues a
    # smooth gradient through it instead of replacing it with a block.
    radius = grow + speck_size // 2 + 1
    kernel = _gaussian_kernel(radius, radius * 0.6)
    keep = 1.0 - mask
    filled = _conv2d_per_channel((x * keep).movedim(1, -1), kernel).movedim(-1, 1)
    weight = _conv2d_per_channel(keep.movedim(1, -1), kernel).movedim(-1, 1)
    filled = filled / weight.clamp_min(1e-3)
    mask = _conv2d_per_channel(mask.movedim(1, -1), _gaussian_kernel(2, 1.0)).movedim(
        -1, 1
    )
    return x * (1.0 - mask) + filled * mask, mask


class DespeckleImage(BaseImage):
    """Remove isolated bright specks (stray white dots) without touching
    hair strands, highlights or other detail.

    A speck is a pixel brighter than its surroundings by `threshold` (white
    top-hat with a `speck_size` square) that belongs to a blob fitting in a
    window twice `speck_size`, in a patch holding few other such pixels. The
    rules that keep lines, textures, eye highlights and fabric sheen are in
    `_speck_mask`. The specks and their dark halo are filled with a weighted
    average of the pixels around them.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "threshold": (
                "FLOAT",
                {
                    "default": 0.08,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How much brighter than its surroundings a pixel must be to count as a speck.",
                },
            ),
            "speck_size": (
                "INT",
                {
                    "default": 5,
                    "min": 3,
                    "max": 31,
                    "step": 2,
                    "tooltip": "Largest speck to remove, in pixels. Anything this wide or wider (an eye highlight) is kept.",
                },
            ),
        }
    }
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "EnhancePack/Image"

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        threshold: float = 0.08,
        speck_size: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = image.device
        image = image.to(comfy.model_management.get_torch_device())
        has_alpha = image.shape[-1] == 4
        alpha = image[..., 3:] if has_alpha else None
        x = image[..., :3].movedim(-1, 1)  # (B, C, H, W)

        mask = _speck_mask(x, threshold, speck_size, 4 * speck_size + 1, MAX_PIXELS)
        rgb, mask = _fill(x, mask, threshold, speck_size, speck_size // 2 + 1)
        rgb = rgb.movedim(1, -1)
        out = torch.cat([rgb, alpha], dim=-1) if has_alpha else rgb
        return (out.to(device), mask[:, 0].to(device))
