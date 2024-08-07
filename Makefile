quality_check:
	isort train_batch.py
	black train_batch.py
	pylint train_batch.py || true

testing: quality_check
	pytest -p no:warnings unit_test.py
	pytest -p no:warnings integration_test.py

build: testing
	docker-compose build train-batch 

run: build
	docker-compose run train-batch 
