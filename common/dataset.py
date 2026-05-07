import json
import os
import numpy as np

_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')


class Dataset:
    """Loads and preprocesses a MONRP benchmark instance from JSON.

    Supports the standard UCLM-SIMD normalized JSON format.
    """

    def __init__(self, name: str):
        self.name = name
        path = os.path.join(_DATASET_DIR, f"{name}.json")
        with open(path, 'r') as f:
            data = json.load(f)

        self.costs: np.ndarray = np.array(data['pbis_cost'], dtype=float)
        self.importances: np.ndarray = np.array(data['stakeholders_importances'], dtype=float)
        # shape: (num_stakeholders, num_requirements)
        self.priorities: np.ndarray = np.array(data['stakeholders_pbis_priorities'], dtype=float)

        raw_deps = data.get('dependencies', None)
        if raw_deps:
            self.dependencies = [list(d) if d is not None else [] for d in raw_deps]
        else:
            self.dependencies = None

        self.num_requirements: int = len(self.costs)
        self.num_stakeholders: int = len(self.importances)

        # Precomputed normalisation denominators
        self.total_cost: float = float(np.sum(self.costs)) or 1.0
        # max possible satisfaction each stakeholder can receive
        self.max_sat_per_stakeholder: np.ndarray = np.sum(self.priorities, axis=1).astype(float)
        self.max_sat_per_stakeholder[self.max_sat_per_stakeholder == 0] = 1.0
        self.total_importance: float = float(np.sum(self.importances)) or 1.0
