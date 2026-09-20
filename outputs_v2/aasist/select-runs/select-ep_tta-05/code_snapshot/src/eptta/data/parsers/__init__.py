"""Dataset plugin registry. Plugins advertise candidates; reviewed contracts drive parsing."""
from importlib import import_module

MODULES = {
    "asvspoof2019_la": "asvspoof2019",
    "asvspoof2021_df": "asvspoof2021",
    "asvspoof2021_la": "asvspoof2021",
    "wavefake": "wavefake",
    "codecfake_xie": "codecfake_xie",
    "in_the_wild": "in_the_wild",
}


def get_dataset_plugin(dataset_id):
    if dataset_id not in MODULES:
        raise KeyError(dataset_id)
    module = import_module(f"eptta.data.parsers.{MODULES[dataset_id]}")
    return module.plugin(dataset_id)


__all__ = ["get_dataset_plugin"]
