from eptta.data.parsers.base import CandidatePlugin


def plugin(dataset_id="codecfake_xie"):
    return CandidatePlugin(dataset_id, ("delimited", "json_records", "sidecar"),
                           ("Xie official and source_authenticity policies remain distinct.",))
