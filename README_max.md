# Getting things to work on my end

## 1. Environment setup

Poetry is used for dependency management. `pyproject.toml` specifies the dependencies needed to run the code.

To create a virtual environemnt to reproduce this code, run `poetry install`. This will install the dependencies in a new vitrualenv and you can now run scripts with `poetry run python your_script.py`.

## 2. Reproducing the paper code

> It is my undertsanding that with the data in https://zenodo.org/record/7714885, one should be able to rerun the "core code" in `create_correlations_file.py`/`plotting_script_*.py` and reproduce the analysis. But I can't.

Issues:

- `create_correlations_file.py` expects a somewhat different naming/placement of the data files.
- `plotting_script_facets.py` expects `qs.csv` and `qsb.csv` in the GHMs folders.
- `plotting_script_lines.py`and `*_maps.py` expects `greenland.csv.`

> I wrote to Dr. Sebastian Gnann regarding this and he provided some `.csv` files which I used to compare the published files with.
> 
> From this comparison, it seems that the published files are the final ones, but some scripts still referenced them wrongly. Therefore, a small modification of the scripts will be needed.