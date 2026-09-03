def get_vocab(model, prompt: list[str]):

	init_token_ids = model.encode(prompt)[0].tolist()
	logits = model.get_logits_from_input_ids(init_token_ids)
	print(logits,"\n\n")
