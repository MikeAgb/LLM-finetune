def get_trainable_parameters(model):
	trainable_params = 0
	all_params = 0

	for _, param in model.named_parameters():
		all_params += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()

	return trainable_params, all_params
