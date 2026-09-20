from eptta.data.parsers.base import CandidatePlugin


def plugin(dataset_id="asvspoof2019_la"):
    return CandidatePlugin(dataset_id, ("official_asvspoof_candidate", "delimited", "sidecar"),
                           ("Official and user-converted protocols must be selected explicitly.",
                            "No column index or label direction is implied by this plugin."))
