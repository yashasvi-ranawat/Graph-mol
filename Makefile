tag = graph_mol

docker-run:
	docker run --runtime nvidia --gpus all -p 8000:8000 $(tag)

docker-build:
	docker build --no-cache -t $(tag) .

fetch-data:
	echo $@
	mkdir -p data/raw
	cd data/raw; curl -O https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/3195389/dsgdb9nsd.xyz.tar.bz2; tar -xvf dsgdb9nsd.xyz.tar.bz2

prepare-data: fetch-data setup-env
	echo $@
	cd data; rm -r processed; mkdir processed
	pipenv run python src/data/prepare_data.py
	cd data; rm -r raw; mkdir raw

jupyter: setup-env
	pipenv run jupyter lab

train: setup-env
	pipenv run python src/training.py

black: setup-env
	pipenv run black --exclude="(processed|raw)" .

test: setup-env
	cd tests; pipenv run pytest .

setup-env: Pipfile.lock
	pipenv sync --dev
