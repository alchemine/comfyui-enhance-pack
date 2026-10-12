"""The ComfyUI modules the pack imports are stubbed so it imports outside
ComfyUI; nothing else is faked.
"""

import sys
import types
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PACK_DIR = Path(__file__).resolve().parent.parent
PACK_NAME = "enhance_pack"


def common_upscale(samples, width, height, upscale_method, crop):
    """`comfy.utils.common_upscale` reduced to the resize the nodes rely on."""
    mode = {
        "nearest-exact": "nearest-exact",
        "bilinear": "bilinear",
        "area": "area",
    }.get(upscale_method, "bicubic")
    return F.interpolate(samples, size=(height, width), mode=mode)


@pytest.fixture(scope="session")
def comfy_stubs():
    utils = types.ModuleType("comfy.utils")
    utils.common_upscale = common_upscale
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")
    comfy = types.ModuleType("comfy")
    comfy.utils = utils
    comfy.model_management = model_management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils
    sys.modules["comfy.model_management"] = model_management


@pytest.fixture(scope="session")
def pack(comfy_stubs):
    """Imports a node module by name, the way ComfyUI would."""
    package = types.ModuleType(PACK_NAME)
    package.__path__ = [str(PACK_DIR)]
    sys.modules.setdefault(PACK_NAME, package)
    return lambda name: importlib.import_module(f"{PACK_NAME}.nodes.{name}")


@pytest.fixture(scope="session")
def mappings(comfy_stubs):
    """The pack's `__init__.py`, loaded the way ComfyUI loads a custom node."""
    spec = importlib.util.spec_from_file_location(
        PACK_NAME, PACK_DIR / "__init__.py", submodule_search_locations=[str(PACK_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACK_NAME] = module
    spec.loader.exec_module(module)
    return module.NODE_CLASS_MAPPINGS
