# Overview of using SENTRA with SGX

## General architecture

SENTRA is exeucted in SGX by using the Occlum framework (https://occlum.io/). Therefore a Docker image will be created which contains the Occlum framework and an occlum instance of the SENTRA code/environment.

## Files

- `docker-compose.yaml` 
Docker compose file which can be used to build and execute SENTRA
- `SentraSGX.dockerfile`
Build file for the docker image. It is based on the Occlum docker images and extend them with the SENTRA code. Thereby the needed Python runtime environment is installed using `miniconda`.
- `entrypoint.sh`
Script which is executed at start of the container. It acually executes SENTRA.
- `sentra-sbom.yaml`
Declarative configuration file which describes which files should be copied into the Occlum instance, e.g. be available in the enclave. This is basically the needed Python runtime environment and the SENTRA code itself.

## Runing SENTRA in SGX

### Requirements

- machine with Intel-SGX enabled 
- recent Linux distribution, which proviedes access to SGX
- recent Docker version installed

### Running SENTRA

Execute:

``docker compose up --profile PROFLE``

This will automatically build the Docker image (if it does not exist) and execute it afterwards. You should see relevant output on the terminal. Note that you need to specify the profile you want to execute:

- ``docker compose up --profile test``
Runs: ``run_training.py``
<br>

- ``docker compose up --profile dp-test``
Runs: ``run_dp_training.py``

- ``docker compose up --profile multi-node``
Runs a multi node scenario with 5 nodes.
