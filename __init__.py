"""Custom nodes mappings."""

from .nodes.image import AdjustImage


NODE_CLASS_MAPPINGS = {
    # FilterPack/Image ###############################################################
    "AdjustImage": AdjustImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    # FilterPack/Image ###############################################################
    "AdjustImage": "Adjust Image",
}
