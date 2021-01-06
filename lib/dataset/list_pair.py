import torch

class PairList:
    def __init__(self, feat):
        device = feat.device if isinstance(feat, torch.Tensor) else torch.device("cpu")
        self.features = feat
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, feat):
        for k, v in feat.extra_fields.items():
            self.extra_fields[k] = v

    # Tensor-like methods
    def to(self, device):
        feat = PairList(self.features.to(device))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            feat.add_field(k, v)
        return feat

    def __getitem__(self, item):
        feat = PairList(self.features[item])
        for k, v in self.extra_fields.items():
            feat.add_field(k, v[item])
        return feat

    def __len__(self):
        return self.features.shape[0]

    def copy_with_fields(self, fields, skip_missing=False):
        feat = PairList(self.features)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                feat.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return feat

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_feats={})".format(len(self))
        return s