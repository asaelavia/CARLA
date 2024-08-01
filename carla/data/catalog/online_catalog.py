from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from carla.data.catalog import DataCatalog
from carla.data.load_catalog import load

from .load_data import load_dataset


class OnlineCatalog(DataCatalog):
    """
    Implements DataCatalog using already implemented datasets. These datasets are loaded from an online repository.

    Parameters
    ----------
    data_name : {'adult', 'compas', 'give_me_some_credit', 'heloc'}
        Used to get the correct dataset from online repository.

    Returns
    -------
    DataCatalog
    """

    def __init__(
            self,
            data_name: str,
            scaling_method: str = "MinMax",
            encoding_method: str = "OneHot_drop_binary",
    ):
        catalog_content = ["continuous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = load(  # type: ignore
            "data_catalog.yaml", data_name, catalog_content
        )
        self.catalog['immutable'] = []
        for key in ["continuous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        # Load the raw data
        raw, train_raw, test_raw = load_dataset(data_name)

        super().__init__(
            data_name, raw, train_raw, test_raw, scaling_method, encoding_method
        )

    @property
    def categorical(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continuous(self) -> List[str]:
        return self.catalog["continuous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]


class AdultCatalog(DataCatalog):
    """
    Implements DataCatalog using already implemented datasets. These datasets are loaded from an online repository.

    Parameters
    ----------
    data_name : {'adult', 'compas', 'give_me_some_credit', 'heloc'}
        Used to get the correct dataset from online repository.

    Returns
    -------
    DataCatalog
    """

    def __init__(
            self,
            data_name: str,
            scaling_method: str = "MinMax",
            encoding_method: str = "OneHot_drop_binary",
    ):
        catalog_content = ["continuous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = {}

        # self.catalog['immutable'] = []
        self.catalog['immutable'] = ['age', 'race', 'sex']
        self.catalog['continuous'] = ['age', 'edunum', 'hours_per_week']
        self.catalog['categorical'] = ['workclass', 'marital_status', 'occupation', 'native_country', 'sex', 'race',
                                       'relationship','education']
        self.catalog['target'] = 'label'
        for key in ["continuous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []
        # Load the raw data
        # raw, train_raw, test_raw = load_dataset(data_name)
        raw = pd.read_csv('adult_clean.csv')
        # for col in raw.columns:
        #     if col in self.continuous + ['label']:
        #         continue
        #     raw[col] = raw[col].astype('object')
        raw[self.catalog['categorical']] = raw[self.catalog['categorical']].astype('category')
        raw[self.catalog['categorical']] = raw[self.catalog['categorical']].apply(lambda x: x.cat.codes)
        y = raw['label']
        train_raw, test_raw, y_train, y_test = train_test_split(raw,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=y)
        super().__init__(
            data_name, raw, train_raw, test_raw, scaling_method, encoding_method
        )

    @property
    def categorical(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continuous(self) -> List[str]:
        return self.catalog["continuous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]
