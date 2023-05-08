# Latent Aggregation

<p align="center">
    <a href="https://github.com/crisostomi/latent-aggregation/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/crisostomi/latent-aggregation/Test%20Suite/main?label=main%20checks></a>
    <a href="https://crisostomi.github.io/latent-aggregation"><img alt="Docs" src=https://img.shields.io/github/deployments/crisostomi/latent-aggregation/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.2.3-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Aggregating seemingly different latent spaces.

## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git@github.com:crisostomi/latent-aggregation.git
cd latent-aggregation
conda env create -f env.yaml
conda activate la
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
```
We use HuggingFace Datasets throughout the project; assuming you already have a HF account (create one if you don't), you will have to login via
```
huggingface-cli login
```
which will prompt you to either create a new token or paste an existing one.

### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```

## Experiment flow

Each experiment `exp_name` in `part_shared_part_novel, same_classes_disj_samples, totally_disjoint` has three scripts:
- `prepare_data_${exp_name}.py` divides the data in tasks according to what the experiment expects;
- `run_${exp_name}.py` trains the task-specific models and uses them to embed the data for each task;
- `analyze_${exp_name}.py` obtains the results for the experiment.

Each script has a corresponding conf file in `conf/` with the same name.
So, to run the `part_shared_part_novel`, you have to first configure the experiment in `conf/prepare_data_part_shared_part_novel.yaml`. In this case, you have to choose a value for `num_shared_classes` and `num_novel_classes_per_task`. Now you will prepare the data via 
```
python src/la/scripts/prepare_data_part_shared_part_novel.py
```
this will populate the `data/${dataset_name}/part_shared_part_novel/` folder. Then you'll embed the data by running
```
python src/la/scripts/run_part_shared_part_novel.py
```
so that now you will have the encoded data in `data/${dataset_name}/part_shared_part_novel/S${num_shared_classes}_N${num_novel_classes_per_task}`.
Having all the latent spaces, you can now run the actual experiment and collect the results by running
```
python src/la/scripts/analyze_part_shared_part_novel.py
```
The results can now be found in `results/part_shared_part_novel`.
