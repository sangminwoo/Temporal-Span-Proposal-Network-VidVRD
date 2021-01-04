import math

import numpy as np
import torch
from torch import nn


class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes=(4, 8, 16),
        anchor_stride=8
    ):
        super(AnchorGenerator, self).__init__()

        cell_anchors = [
            generate_anchors(anchor_stride, sizes).float()
        ]
        self.stride = anchor_stride
        self.cell_anchors = BufferList(cell_anchors)

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, time_width):
        anchors = []
        for base_anchors in self.cell_anchors:
            device = base_anchors.device
            shifts = torch.arange(
                0, time_width+1, step=self.stride, dtype=torch.float32, device=device
            )
            anchors.append(
                (shifts.view(-1, 1, 1) + base_anchors.view(1, -1, 1)).reshape(-1, 2)
            )

        return anchors

    def forward(self, rel_feats):
        time_width = rel_feats.shape[2] # NxCxT (time dimension)
        anchors = self.grid_anchors(time_width)
        return anchors
    

def generate_anchors(
	    stride=8, sizes=(4, 8, 16)
	):
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride
    )


def _generate_anchors(stride, sizes):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    sizes wrt a reference (stride - 1) window.
    """
    anchor = np.array([0, stride], dtype=np.float)
    anchors = _scale_enum(anchor, sizes)
    return torch.from_numpy(anchors)


def _scale_enum(anchor, sizes):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    ctr, width = anchor[0], anchor[1]
    ws = width * sizes
    anchors = _mkanchors(ws, ctr)
    return anchors


def _mkanchors(ws, ctr):
    """Given a vector of widths (ws) around a center (ctr),
    output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    anchors = np.hstack(
        (
            ctr - 0.5 * ws,
            ctr + 0.5 * ws,
        )
    )
    return anchors


def make_anchor_generator(param):
    anchor_sizes = param['anchor_sizes']
    anchor_stride = param['anchor_stride']
    assert len(anchor_stride) == 1, "should have a single ANCHOR_STRIDE"

    anchor_generator = AnchorGenerator(anchor_sizes, anchor_stride)
    return anchor_generator


if __name__=='__main__':
	rel_feats = torch.randn(2,4,60) # NxCxT
	anchor_sizes = (15, 30, 45, 60)
	anchor_stride = 7.5
	anchor_generator = AnchorGenerator(anchor_sizes, anchor_stride)
	anchors = anchor_generator(rel_feats)
	print(anchors)
	print(anchor_generator.num_anchors_per_location()[0])