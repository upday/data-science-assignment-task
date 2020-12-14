train:
	docker-compose run --rm notebook /bin/bash -c "python3 model_train.py"

evaluate:
	docker-compose run --rm notebook /bin/bash -c "python3 model_evaluate.py $(DATASET)"

predict:
	docker-compose run --rm notebook /bin/bash -c "python3 model_predict.py $(DATASET)"

.PHONY: train evaluate predict