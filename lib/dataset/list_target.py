import torch

class TargetList:
    def __init__(self, target):
        device = target.device if isinstance(target, torch.Tensor) else torch.device("cpu")
        self.target = target
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, target):
        for k, v in target.extra_fields.items():
            self.extra_fields[k] = v

    # Tensor-like methods
    def to(self, device):
        target = TargetList(self.target.to(device))
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            target.add_field(k, v)
        return target

    def __getitem__(self, item):
        target = TargetList(self.target[item])
        for k, v in self.extra_fields.items():
            target.add_field(k, v[item])
        return target

    def __len__(self):
        return self.target.shape[0]

    def copy_with_fields(self, fields, skip_missing=False):
        target = TargetList(self.target)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                target.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return target

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_targets={})".format(len(self))
        return s