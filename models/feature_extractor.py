# ============================================================
# Intermediate Layer Feature Extractor
# ============================================================

import torch


class FeatureExtractor:

    """
    Extract intermediate layer features
    from Vision Transformer
    """

    def __init__(self, model):

        self.model = model

        self.features = {}

        self._register_hooks()


    def _get_hook(self, name):

        """
        Hook function
        """

        def hook(module, input, output):

            self.features[name] = output.detach()

        return hook


    def _register_hooks(self):

        """
        Register hooks to transformer blocks
        """

        # Shallow layer
        self.model.blocks[0].register_forward_hook(

            self._get_hook("layer_1")
        )

        # Middle layer
        self.model.blocks[5].register_forward_hook(

            self._get_hook("layer_6")
        )

        # Deep layer
        self.model.blocks[11].register_forward_hook(

            self._get_hook("layer_12")
        )


    def extract_features(self, images):

        """
        Extract intermediate features
        """

        self.features = {}

        self.model.eval()

        with torch.no_grad():

            _ = self.model(images)

        return self.features
