"""Utils for whole project"""

from pathlib import Path
from typing import Union, Dict, Any

import yaml


def load_config(path: Union[Path, str]) -> Dict[str, Any]:
    path = Path(path)
    
    with path.open(mode='r', encoding='UTF-8') as file:
        config = yaml.load(file.read(), Loader=yaml.Loader)

    return config
