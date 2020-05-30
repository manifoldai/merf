.PHONY: help, ci-black, ci-flake8, ci-test, isort, black, docs

PROJECT=merf
CONTAINER_NAME="merf_jupyter_${USER}"  ## Ensure this is the same name as in docker-compose.yml file
VERSION_FILE:=VERSION
TAG:=$(shell cat ${VERSION_FILE})

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

release-pypi:
	echo "Not implemented yet"

git-tag:  ## Tag in git, then push tag up to origin
	git tag $(TAG)
	git push origin $(TAG)

ci-black: ## Test for black requirements
	docker exec $(CONTAINER_NAME) black --check merf

ci-flake8: ## Test for flake8 requirements
	docker exec $(CONTAINER_NAME) flake8 merf

ci-test:  ## Test unittests
	docker exec $(CONTAINER_NAME) python merf/merf_test.py -v

ci: ci-black ci-flake8 ci-test ## Check black, flake8, and unittests
	@echo "CI sucessful"

isort: ## Runs isort, which sorts imports
	docker exec $(CONTAINER_NAME) isort -rc merf

black: ## Run black, which formats code
	docker exec $(CONTAINER_NAME) black merf

lint: isort black ## Lint repo; runs black and isort on all files
	@echo "Linting complete"

dev-start: ## Primary make command for devs, spins up containers
	@echo "Building new images from compose"
	docker-compose -f docker/docker-compose.yml --project-name $(PROJECT) up -d --build

dev-stop: ## Spins down active containers
	docker kill $(CONTAINER_NAME)

docs: ## Creates docs
	docker exec -e GRANT_SUDO=yes $(CONTAINER_NAME) bash -c "cd docsrc; make html"
	@cp -a docsrc/_build/html/. docs