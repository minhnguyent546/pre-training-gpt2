CONFIG_FILE=config/config.yaml
INFERENCE_DIR=/data/transformers/gpt2-2024-05-16-05-30/storage/checkpoints/
INFERENCE_CHECKPOINT=$(INFERENCE_DIR)/gpt2-20000.pt
INFERENCE_TOKENIZER=$(INFERENCE_DIR)/tokenizer.json

.PHONY: default preprocess train compress generate

default:
	@echo "GPT2 model"
	@echo "Run \"make preprocess\" to preprocess data and \"make train\" to train the model"

preprocess:
	python gpt2/preprocess.py -c $(CONFIG_FILE)

train:
	python gpt2/train.py -c $(CONFIG_FILE)

compress:
	cd ..; tar -czvf gpt2-from-scratch{.tar.gz,}

generate:
	python gpt2/generate.py \
			--model $(INFERENCE_CHECKPOINT) \
			--tokenizer $(INFERENCE_TOKENIZER) \
			--seed 42 \
			--max-new-tokens 100 \
			--temperature 0.8
