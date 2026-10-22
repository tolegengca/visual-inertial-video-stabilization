from __future__ import annotations

import cv2
import numpy as np


class FlowEstimator:
    """Bidirectional dense flow using pretrained RAFT-Small or Farneback fallback."""

    def __init__(self, backend="raft", device="auto", working_width=480):
        self.backend, self.working_width = backend, working_width
        self.model = self.transforms = None
        if backend == "raft":
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch, self.device = torch, torch.device(device)
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=True).to(self.device).eval()
            self.transforms = weights.transforms()

    @staticmethod
    def _resize(frame, width):
        h, w = frame.shape[:2]
        if not width or w <= width:
            return frame, 1.0
        scale = width / w
        return cv2.resize(frame, (width, round(h * scale)), interpolation=cv2.INTER_AREA), scale

    def _one_raft(self, a, b):
        torch = self.torch
        a, scale = self._resize(a, self.working_width)
        b, _ = self._resize(b, self.working_width)
        h, w = a.shape[:2]
        ph, pw = (-h) % 8, (-w) % 8
        def tensor(x):
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            return torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
        x1, x2 = self.transforms(tensor(a), tensor(b))
        with torch.inference_mode():
            flow = self.model(x1.to(self.device), x2.to(self.device))[-1][0]
        flow = flow[:, :h, :w].cpu().permute(1, 2, 0).numpy()
        return flow, scale

    @staticmethod
    def _one_farneback(a, b):
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(ga, gb, None, .5, 4, 25, 4, 7, 1.5, 0)

    def bidirectional(self, a, b):
        if self.backend == "raft":
            f, scale = self._one_raft(a, b)
            r, _ = self._one_raft(b, a)
            return f, r, scale
        a, scale = self._resize(a, self.working_width)
        b, _ = self._resize(b, self.working_width)
        return self._one_farneback(a, b), self._one_farneback(b, a), scale

