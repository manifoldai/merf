# Development Workflow

This is a short README about how to do development in this repo. 

## Git Feature Branch Workflow

We have protected the `master` branch of the repo so that you cannot push directly to it. Rather you should make a feature branch, e.g. `sdey/<feature_name>`, do all your work there, and then do a pull request into master from there. 

For more information about git and the feature branching workflow, please see the resources below. 

* https://www.atlassian.com/git
* https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow

## Continuous Integration

In order to keep the quality of the code high, you should setup continuous integration (CI) on this repo. The CI jobs should be run automatically every time code is pushed to a branch. You should not be able to merge to master unless CI passes. 

A basic CI job should check for two things: 
* Code passes lint.  Specifically it checks that PEP8 standards are met using `black` and `flake8`.
* The unit tests pass.  Specifically it uses `pytest` to run the unit tests that are defined through out the repo. 

You can run CI locally by running the script:
```bash
./scripts/docker/ci.sh
```

### Lint

[Linting](https://realpython.com/python-code-quality/) -- or checking the code style -- is an important part of any software development practice.  To make things simple, we have created a script that automates linting as much as possible. It can be invoked (from inside the Docker container as always):
```bash
./scripts/docker/autoformat.sh
```

This should use black and isort to automatically format your code so that it passes CI. If it is unable to, it will let you know. You can check if CI passes after auto-formatting by running the `ci.sh` script above. 

### Unit Tests

We have written a number of unit tests for this repo. You can run all of them using `pytest`.  You can run a specific test suite by running that specific python file:
```bash
pytest orbyter_demo/scripts/evaluate_test.py -v
```

You can even just run a specific test by issuing a command like: 
```bash
pytest orbyter_demo/scripts/evaluate_test.py::test_evaluate -v
```

Read more about pytest and how to invoke tests [here](https://docs.pytest.org/en/latest/usage.html). 


## MLFlow

We are using MLFlow for experiment tracking. MLFlow stores parameters, metrics, artifacts, and even models.
Experiments are grouped by their experiment_name, which is set in the config.yml. We are
using it primarily to store model parameters, model metrics, and artifacts like input data and config files.

There are two primary types of storage/tracking: 1) tracking URI and 2) Artifact storage. URI tracking can be done
locally or in a SQL database (e.g., postgreSQL), and artifact storage can be done locally or in the cloud (e.g., S3).
Parameters (`mlflow.log_params()`), metrics (`mlflow.log_metrics()`), git commit (done automatically), run times
(done automatically), and source file (done automatically) are stored in the tracking URI. File artifacts
(`mlflow.log_artifacts`) are stored in artifact storage.

For our projects, we will be operating in one of two modes: 1) all local or 2) all cloud. These modes are set
by the MLFLOW_TRACKING_URI and MLFLOW_ARTIFACT_LOCATION in the .env. This file is passed to
the docker compose. *Do not add quotes around the parameter values in the .env or the quotes will be
included in the parameter value in the environment, and things will get messed up*.

MLFlow's UI is running on the mlflow docker container. See what port it's mapped to using `docker ps`, and go to
`localhost:xxx`. For example, if `docker ps` shows:

```bash
CONTAINER ID        IMAGE                          COMMAND                  CREATED             STATUS              PORTS                       NAMES
f168e19b8b67        orbyter_demo_mlflow            "bash -c 'mlflow ui …"   4 days ago          Up 3 days           127.0.0.1:32770->5000/tcp   orbyter_demo_mlflow_<username>
87f03baf686e        orbyter_demo_bash-executer     "/bin/bash"              4 days ago          Up 4 days           127.0.0.1:32768->8501/tcp   orbyter_demo_sdey_bash-executer_<username>
d9bd01600486        orbyter_demo_notebook-server   "bash -c 'cd /mnt &&…"   4 days ago          Up 3 days           127.0.0.1:32769->8888/tcp   orbyter_demo_sdey_notebook-server_<username>
```

Then go to `localhost:32770` on your browser to access the MLFlow UI.

### Local

For local storage, MLFLOW_TRACKING_URI is set to a local path and MLFLOW_ARTIFACT_LOCATION is set to None, i.e., left blank.
For example, to set the tracking URI to the local directory /mnt/experiments, in the .env

```
MLFLOW_TRACKING_URI=/mnt/experiments
MLFLOW_ARTIFACT_LOCATION=
```

When setting the MLFLOW_ARTIFACT_LOCATION to None (blank), mlflow will log artifacts in the same location at the tracking URI.
Issues can arise when you set them to the same value so just leave it blank. For local runs, different experiments
(set by experiment_name in config.yml) are grouped under different directories.

### Remote

When doing experiment remotely, we are using postgreSQL for the tracking URI and S3 for the artifacts.
Enter the postgres URL, s3 path, and AWS credentials in the .env file.

```
MLFLOW_TRACKING_URI=postgresql://username:password@url:port/database
MLFLOW_ARTIFACT_LOCATION=s3://my_bucket/
AWS_ACCESS_KEY_ID={your_access_key}
AWS_SECRET_ACCESS_KEY={your_secret_key}
```

If you originally started with local storage and switch to remote tracking, you should rebuild your docker containers so
the environment variables are correctly loaded into the container. If you do not, MLFlow's UI will not be looking
in the correct location. To do this, exit the container and run:

```bash
./scripts/start.sh
```

### Important note on new experiments with different storage locations

A single experiment is either local or cloud -- it can't be both. If you start a local experiment, it will always be local.
Same for the cloud.
By running a new experiment (by changing the experiment's name), you can change the logging location.
All experiments with the same name are logged to the same place. If you want to change the location of logging, update MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_LOCATION, _and_ the experiment name in the config.yml.

## Adding new libraries

The Dockerfile also installs any project specific libraries from docker/requirements.txt. When you want to add a new library,
add it to the requirements.txt files, e.g, my_lib==x.x.x. 

```bash
./scripts/start.sh
```

Or build an image directly, go to the docker folder `cd docker`, and run

```bash
docker build -t image_name:tag .
```

Where image_name and tag are whatever you want to name and tag your image.

## Auto-documentation with sphinx

To build the documentation for the project, 

```bash
./scripts/make_docs.sh
```

You can open the index.html to read the docs, which is located in `docs/_build/html/index.html`.

When adding new modules, make sure docs/index.rst is up to date, i.e., the new modules are added.

## Logging

All logging configurations are done through the logging config, logging.yml. The base handlers (where logging outputs go) are the console, an info 
file, and an error file. They all have baseline message severity levels as well as formats. You can also change logging at the
module level, under loggers. Note, you can only increase the severity of the messages at the logger level. The root is kind of like the loggers, but 
it applies to the root logger, i.e., the module executed from the command line.


## Common pitfalls

### 1. Local tests passing, CI failing

Make sure your cloud CI base image is pointing to the same base image as the 
docker container. Ideally, the cloud CI image should be the exact same as the working image, which
can be done by registering your image in your cloud CI.

### 2. Mismatch between mlflow logging and UI

This can happen when the .env is updated, but the image has not been rebuilt.
Exit the container and rebuild with `./scripts/start.sh`