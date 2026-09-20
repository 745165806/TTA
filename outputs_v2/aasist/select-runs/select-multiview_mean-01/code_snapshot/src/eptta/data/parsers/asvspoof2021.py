from eptta.data.parsers.base import CandidatePlugin


def plugin(dataset_id):
    return CandidatePlugin(dataset_id, ("official_asvspoof_candidate", "delimited", "json_records", "sidecar"),
                           ("LA and DF keys are distinct contracts.", "Keys without labels remain unlabeled."))
