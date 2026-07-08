"""Material -> friction tier lookup, loaded from a YAML file at startup."""

import yaml


class FrictionMap:
    def __init__(self, path):
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        self._table = {str(k).lower(): str(v).lower() for k, v in raw.items()}

    def materials(self):
        return list(self._table.keys())

    def to_friction(self, material, default="unknown"):
        return self._table.get(str(material).lower(), default)
