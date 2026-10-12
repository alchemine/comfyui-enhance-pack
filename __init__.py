"""Custom nodes mappings."""

from .nodes.image import AdjustImage


NODE_CLASS_MAPPINGS = {
    # EnhancePack/Image ###############################################################
    "AdjustImage": AdjustImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    # EnhancePack/Image ###############################################################
    "AdjustImage": "Adjust Image",
}
