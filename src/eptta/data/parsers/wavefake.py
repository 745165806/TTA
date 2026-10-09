from eptta.data.parsers.base import CandidatePlugin


def plugin(dataset_id="wavefake"):
    return CandidatePlugin(dataset_id, ("embedded_parquet",),
                           ("Local label values and audio_id grouping were reviewed; full audio import remains unrun.",))
