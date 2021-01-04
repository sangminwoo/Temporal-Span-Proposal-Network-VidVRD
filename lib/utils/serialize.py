from collections import OrderedDict

def load_checkpoint(model, checkpoint):
	model_state_dict = model.state_dict()
	
	try:
		model.load_state_dict(checkpoint)
	except:
		new_checkpoint = OrderedDict()

		model_keys = [key for key in model.state_dict().keys()]
		checkpoint_keys = [key for key in checkpoint.keys()]
		if 'module.' not in model_keys[0] and 'module.' in checkpoint_keys[0]:
			for key, value in checkpoint.items():
				new_key = key[7:] # remove "module."
				new_checkpoint[new_key] = value
		elif 'module.' in model_keys[0] and 'module.' not in checkpoint_keys[0]:
			for key, value in checkpoint.items():
				new_key = 'module.' + key
				new_checkpoint[new_key] = value
		else:
			raise ValueError('failed to load checkpoint')

		model.load_state_dict(new_checkpoint)
	# return model