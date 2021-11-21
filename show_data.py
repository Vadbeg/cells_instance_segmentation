"""CLI for showing annotated data"""

import warnings

import typer

from cells_instance_segmentation.cli.show_data_cli import show_data


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    typer.run(show_data)
