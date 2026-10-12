"""Custom nodes mappings."""

from .nodes.image import AdjustImage, DespeckleImage


NODE_CLASS_MAPPINGS = {
    # EnhancePack/Image ###############################################################
    "AdjustImage": AdjustImage,
    "DespeckleImage": DespeckleImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    # EnhancePack/Image ###############################################################
    "AdjustImage": "Adjust Image",
    "DespeckleImage": "Despeckle Image",
}
