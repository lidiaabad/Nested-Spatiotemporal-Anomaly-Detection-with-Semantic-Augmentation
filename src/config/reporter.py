import os
import json
import numpy as np

class Reporter:
    def __init__(self, experiment_path, filename):
        self.experiment_path = experiment_path
        self.file_name = filename
        self.output_file = os.path.join(experiment_path, self.file_name)
        self.data = {}
    
    def _convert_data(self, data):
        if isinstance(data, np.float32):
            return float(data)  # Convert np.float32 to Python float (which is float64)
        elif isinstance(data, dict):
            return {key: self._convert_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_data(item) for item in data]
        return data

    def report(self, key, data):
        self.data[key] = data

    def save(self):
        converted_data = self._convert_data(self.data)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=4)

    def load(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __del__(self):
        # Never crash during interpreter shutdown
        try:
            self.save()
        except Exception:
            pass