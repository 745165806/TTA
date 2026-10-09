from eptta.data.parsers.base import CandidatePlugin


def plugin(dataset_id="dfadd"):
    return CandidatePlugin(dataset_id, ("embedded_arrow",),
                           ("An explicit source-group map and observed label mapping are required.",))
