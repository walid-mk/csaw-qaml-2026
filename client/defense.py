"""Local defense protocol helpers. Never import untrusted submissions in a grader."""
import torch


class DefendedModel(torch.nn.Module):
    """Apply a participant image transform, then the original quantum architecture.

    For non-differentiable transforms, bpda_identity uses the true forward values
    and an identity backward approximation for adaptive attack development.
    """
    def __init__(self, model, transform, *, state=None, config=None, backward='autograd'):
        super().__init__()
        if backward not in ('autograd', 'bpda_identity'):
            raise ValueError('Choose autograd or bpda_identity')
        self.model, self.transform = model, transform
        self.state, self.config, self.backward = state or {}, config or {}, backward

    def forward(self, images):
        transformed = self.transform(images, state=self.state, config=self.config)
        if not isinstance(transformed, torch.Tensor) or transformed.shape != images.shape or transformed.dtype != images.dtype:
            raise ValueError('Defense preprocessing must retain image shape and float64 dtype')
        if not torch.isfinite(transformed).all() or transformed.min() < 0 or transformed.max() > 1:
            raise ValueError('Defense preprocessing must return finite pixels in [0,1]')
        if self.backward == 'bpda_identity':
            transformed = images + (transformed - images).detach()
        return self.model(transformed)
