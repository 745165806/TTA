from eptta.data.parsers.base import CandidatePlugin


def plugin(dataset_id="wavefake"):
    return CandidatePlugin(dataset_id, ("delimited", "json_records", "sidecar"),
                           ("Directory names are not accepted as labels without review.",))
