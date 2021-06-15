# config.py

from pathlib import Path

class Config:
    """ folder constants """"
    #paths
    root = Path.cwd().resolve()
    data = Path(root) / "data"
    notebook = Path( root) / "notebook"
    model = Path( root ) / "model"

    

