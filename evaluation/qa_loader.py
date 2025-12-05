# evaluation/qa_loader.py

import json
from pathlib import Path
from typing import Union, Dict, Any


def load_qa(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load QA dataset from qa_set.json.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
