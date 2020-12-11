train:
	docker-compose run --rm notebook /bin/bash -c "python3 model_train.py"

evaluate:
	docker-compose run --rm notebook /bin/bash -c "python3 model_evaluate.py $(DATASET)"

.PHONY: train