from eptta.data.parsers.base import CandidatePlugin


def plugin(dataset_id="in_the_wild"):
    return CandidatePlugin(dataset_id, ("delimited", "json_records", "sidecar"),
                           ("Metadata columns and collection groups require explicit review.",))
