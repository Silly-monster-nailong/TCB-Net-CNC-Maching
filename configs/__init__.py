# configs/__init__.py
import yaml

class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(data)