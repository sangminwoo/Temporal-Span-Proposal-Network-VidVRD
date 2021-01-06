import numpy as np

class AverageMeter:
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val
		self.count += n
		self.avg = self.sum / self.count


def calculate_eta_iter(one_batch_time, cur_iter, max_iter):
	eta_in_iter = one_batch_time * (max_iter-cur_iter-1)
	return eta_in_iter


def calculate_eta_epoch(one_batch_time, cur_epoch, max_epoch, cur_iter, max_iter):
	eta_in_iter = calculate_eta_iter(one_batch_time, cur_iter, max_iter)
	eta_in_epoch = (one_batch_time * max_iter) * (max_epoch-cur_epoch-1) 
	eta_seconds = eta_in_iter + eta_in_epoch
	return eta_seconds


def normalize(x, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)


def to_onehot(x, num_classes):
    """ one-hot encodes a tensor """
    return np.eye(num_classes, dtype='float32')[x]


def to_multi_onehot(x, num_classes):
	one_hot = np.zeros(num_classes)
	one_hot[x] = 1
	return one_hot

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)