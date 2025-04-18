# NumTabQA: A Benchmark for Analyzing Numerical Skill of Language Models for Table Question Answering

[![Docker Hub](https://img.shields.io/docker/v/konstantinjdobler/nlp-research-template/latest?color=blue&label=docker&logo=docker)](https://hub.docker.com/r/konstantinjdobler/nlp-research-template/tags) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![License: MIT](https://img.shields.io/github/license/konstantinjdobler/nlp-research-template?color=green)

## Dataset Files

The dataset files can be downloaded from following link:

https://drive.google.com/drive/folders/1YvgNTUbFoV6SVxKcb47fUh2gYtopbQAS?usp=drive_link

There is one folder for every table corpus (WTQ and GitTales) and data split (train, validation, test). The table data is stored separately from the question data, respectively (e.g. wikitablequestions_test_tables) to save memory.

To inspect the question data simply load the arrow file with the datasets library:

```python
import datasets
dataset = datasets.Dataset.load_from_disk('/my/local/path/wikitablequestions_test/250205_2114_47_349337')
```

We plan to make our benchmark available through the [Hugging Face Hub](https://huggingface.co/docs/hub/index) with the [datasets API](https://huggingface.co/docs/datasets/load_hub) after publishing our paper.

## Fine-Tuning and Evaluation Models
To fine-tune one of the smaller table models (e.g. TAPEX) go to `scripts/console.sh` specify the GPU IDs of your machine you want to use (e.g. `DEVICES="0,1"`) for cuda:0 and cuda:1. Then start the container by running 
```bash
bash scripts/console.sh
```
The environment within the container is already setup for you so you can simply run the following command in the container:
```python
python train.py
```
You can add arguments (e.g. `--model tapex`) to change the configuration. See a full list with explanation for each argument by running `python train.py —-help` or check out the file `src/numerical_table_questions/arguments.py`.
The defaults correspond to the values used in our experiments.
<details><summary>Using GPUs for hardware acceleration</summary>

<p>

By default, `train.py` already detects all available CUDA GPUs and uses `DistributedDataParallel` training in case multiple GPUs are found. You can manually select specific GPUs with `--cuda_device_ids`. To use different hardware accelerators, use the `--accelerator` flag. You can use advanced parallel training strategies with `--distributed_strategy`.

</p>
</details>

If you need guidance on how to setup Docker please read the section [Setup](#setup) below.

For testing one of the smaller table models (e.g. TAPEX) including your custom fine-tuned models run:
```python
python src/numerical_table_questions/evaluation.py
```
Use the same configuration as during training. Specify the checkpoint of your fine-tuned model with `--checkpoint path/to/saved/checkpoint.ckpt`.

### Evaluating LLMs with lm_eval and vLLM
For evaluating one of the LLMs we make use of the well-established [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework. We have implemented our task in this repository (`lm-evaluation-harness/lm-eval/tasks/num_tab_qa`). After publishing our paper we plan contribute our benchmark as a task in the lm-evaluation-harness package for ease of use.

In our experiments we used a modified container for lm_eval. To build the container image run: 
```bash
bash docker_build_image.sh
```
To start an container instance run (and press any key when prompted):
```
bash docker_lm_eval_container.sh 0,1
```
**Note:** 0,1 selects GPUs 0 and 1 of your machine. If you do not specify any GPU IDs the job will run on CPU which is likely to be very slow (not recommended).

Attach to the container you just created:
```bash
docker attach CONTAINER_ID
```
You can see all containers and lookup the CONTAINER_ID with:
```bash
docker ls -—all
```
Then within the container install some dependencies by running:
```bash
.  docker_lm_eval_environment.sh
```
Everything is setup now and you can use the same container for multiple evaluation runs (one after the other). Simply run:
```bash
bash test_vllm_model.sh
```
You can change the configuration by editing `test_vllm_model.sh`. For an explanation of the arguments see the documentation of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [vLLM](https://docs.vllm.ai/en/latest/).

## <a name="setup"></a> Setup

### Preliminaries

It's recommended to use [`mamba`](https://github.com/mamba-org/mamba) to manage dependencies. `mamba` is a drop-in replacement for `conda` re-written in C++ to speed things up significantly (you can stick with `conda` though). To provide reproducible environments, we use `conda-lock` to generate lockfiles for each platform.

<details><summary>Installing <code>mamba</code></summary>

<p>

On Unix-like platforms, run the snippet below. Otherwise, visit the [mambaforge repo](https://github.com/conda-forge/miniforge#mambaforge). Note this does not use the Anaconda installer, which reduces bloat.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

</details>

<details><summary>Installing <code>conda-lock</code></summary>

<p>

The preferred method is to install `conda-lock` using `pipx install conda-lock`. For other options, visit the [conda-lock repo](https://github.com/conda/conda-lock). For basic usage, have a look at the commands below:

```bash
conda-lock install --name myenv conda-lock.yml # create environment with name myenv based on lockfile
conda-lock # create new lockfile based on environment.yml
conda-lock --update <package-name> # update specific packages in lockfile
```

</details>

### Environment

Lockfiles are an easy way to **exactly** reproduce an environment.

After having installed `mamba` and `conda-lock`, you can create a `mamba` environment named `myenv` from a lockfile with all necessary dependencies installed like this:

```bash
conda-lock install --name myenv conda-lock.yml
```

You can then activate your environment with

```bash
mamba activate myenv
```

To generate new lockfiles after updating the `environment.yml` file, simply run `conda-lock` in the directory with your `environment.yml` file.

For more advanced usage of environments (e.g. updating or removing environments) have a look at the [conda-documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment).

<details><summary>Setup on <code>ppc64le</code></summary>

<p>

**If you're not using a PowerPC machine, do not worry about this.**

Whenever you create an environment for a different processor architecture, some packages (especially `pytorch`) need to be compiled specifically for that architecture. IBM PowerPC machines for example use a processor architecture called <code>ppc64le</code>.
Setting up the environment <code>ppc64le</code> is a bit tricky because the official channels do not provide packages compiled for <code>ppc64le</code>. However, we can use the amazing [Open-CE channel](https://ftp.osuosl.org/pub/open-ce/current/) instead. A lockfile containing the relevant dependencies is already prepared in <code>ppc64le.conda-lock.yml</code> and the environment again can be simply installed with:

```bash
conda-lock install --name myenv-ppc64le ppc64le.conda-lock.yml
```

Dependencies for <code>ppce64le</code> should go into the seperate <code>ppc64le.environment.yml</code> file. Use the following command to generate a new lockfile after updating the dependencies:

```bash
conda-lock --file ppc64le.environment.yml --lockfile ppc64le.conda-lock.yml
```

</p>
</details>

### Docker

For fully reproducible environments and running on compute clusters, we provide pre-built docker images at [konstantinjdobler/nlp-research-template](https://hub.docker.com/r/konstantinjdobler/nlp-research-template/tags). We also provide a `Dockerfile` that allows you to build new docker images with updated dependencies:

```bash
docker build --tag <username>/<imagename>:<tag> --platform=linux/amd64 .
```

The specified username should be your personal [`dockerhub`](https://hub.docker.com) username. This will make distribution and reusage of your images easier.

## Development

Development for ML can be quite resource intensive. If possible, you can start your development on a more powerful host machine to which you connect to from your local machine. Normally, you would set up the correct environment on the host machine as explained above but this workflow is simplified a lot by using `VS Code Dev Containers`. They allow you to develop inside a docker container with all necessary dependencies pre-installed. The template already contains a `.devcontainer` directory, where all the settings for it are stored - you can start right away!

<details><summary>VS Code example</summary>

<p>

After having installed the [Remote-SSH-](https://code.visualstudio.com/docs/remote/ssh), and [Dev Containers-Extension](https://code.visualstudio.com/docs/devcontainers/containers), you set up your `Dev Container` in the following way:

1. Establish the SSH-connection with the host by opening your VS Code command pallet and typing <code>Remote-SSH: Connect to Host</code>. Now you can connect to your host machine.
2. Open the folder that contains this template on the host machine.
3. VS Code will automatically detect the `.devcontainer` directory and ask you to reopen the folder in a Dev Container.
4. Press <code>Reopen in Container</code> and wait for VS Code to set everything up.

When using this workflow you will have to adapt `"runArgs": ["--ipc=host", "--gpus", "device=CHANGE_ME"]` in [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) and specify the GPU-devices you are actually going to use on the host machine for your development. Optionally you can mount cache files with `"mounts": ["source=/MY_HOME_DIR/.cache,target=/home/mamba/.cache,type=bind"]`. `conda-lock` is automatically installed for you in the Dev Container.

Additionally, you can set the `WANDB_API_KEY` in your remote environment; it will then be automatically mapped into the container.

</p>
</details>

### Using the Docker environment for training

To run the training code inside the docker environment, start your container by executing the [console.sh](./scripts/console.sh) script. Inside the container you can now execute your training script as before.

```bash
bash ./scripts/console.sh   # use this to start the container
python train.py -n <run-name> -d /path/to/data/ --model tapex --offline # execute the training inside your container
```

Like when using a [`Dev Container`](#development), by default no GPUs are available inside the container and caches written to `~/.cache` inside the container will not be persistent. You can modify the [console.sh](./scripts/console.sh) script to select GPUs for training, a persistent cache directory and the docker image for the container. Also, make sure to mount the data directory into the container.

**Docker + GPUs:** Always select specififc GPUs via `docker` (e.g. `--gpus device=0,7` for the GPUs with indices `0` and `7` in [console.sh](./scripts/console.sh)) and set the `train.py` script to use all available GPUs for training with `--num_devices=-1` (which is the default).

**Note:** In order to mount a directory for caching you need to have one created first.

<details><summary>Single-line docker command</summary>

<p>

You can start a script inside a docker container in a single command (caches are not persistent in this example):

```bash
docker run -it --user $(id -u):$(id -g) --ipc host -v "$(pwd)":/workspace -w /workspace --gpus device=0,7 konstantinjdobler/nlp-research-template:latest python train.py --num_devices=-1 ...
```

</p>
</details>

<details><summary>Using Docker with SLURM / <code>pyxis</code></summary>

<p>

For security reasons, `docker` might be disabled on your cluster. You might be able to use the SLURM plugin `pyxis` instead like this:

```bash
srun ... --container-image konstantinjdobler/nlp-research-template:latest python train.py ...
```

This uses [`enroot`](https://github.com/NVIDIA/enroot) under the hood to import your docker image and run your code inside the container. See the [`pyxis` documentation](https://github.com/NVIDIA/pyxis) for more options, such as `--container-mounts` or `--container-writable`.

If you want to run an interactive session with bash don't forget the `--pty` flag, otherwise the environment won't be activated properly.

</p>
</details>

### Weights & Biases

Weights & Biases is a platform that provides an easy way to log training results for ML researchers. It lets you create checkpoints of your best models, can save the hyperparameters of your model and even supports Sweeps for hyperparameter optimization. For more information you can visit the [website](https://wandb.ai/site). To enable Weights & Biases, enter your `WANDB_ENTITY` and `WANDB_PROJECT` in [dlib/frameworks/wandb.py](dlib/frameworks/wandb.py) and omit the `--offline` flag for training.

<details><summary>Weights & Biases + Docker</summary>

<p>

When using docker you also have to provide your `WANDB_API_KEY`. You can find your personal key at [wandb.ai/authorize](https://app.wandb.ai/authorize). Either set `WANDB_API_KEY` on your host machine and use the `docker` flag `--env WANDB_API_KEY` when starting your run or use `wandb docker-run` instead of docker run.

</p>
</details>
