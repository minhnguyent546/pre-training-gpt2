CONFIG_FILE=config/config.yaml

.PHONY: default preprocess train compress

default:
	@echo "GPT2 model"
	@echo "Run \"make preprocess\" to preprocess data and \"make train\" to train the model"

preprocess:
	python gpt2/preprocess.py -c $(CONFIG_FILE)

train:
	python gpt2/train.py -c $(CONFIG_FILE)

compress:
	cd ..; tar -czvf gpt2-from-scratch{.tar.gz,}
