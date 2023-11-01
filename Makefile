tag = graph_mol

docker-run:
	docker run --runtime nvidia --gpus all -p 8000:8000 $(tag)

docker-build:
	docker build --no-cache -t $(tag) .

fetch-data:
	echo $@
	cd data/raw; curl -O https://s3-eu-west-1.amazonaws.com/pstorage-npg-968563215/3195389/dsgdb9nsd.xyz.tar.bz2; tar -xvf dsgdb9nsd.xyz.tar.bz2

prepare-data: fetch-data setup-env
	echo $@
	pipenv run python src/data/prepare_data.py
	cd data; rm -r raw; mkdir raw

black: setup-env
	pipenv run black --exclude data .

setup-env: Pipfile.lock
	pipenv sync --dev
