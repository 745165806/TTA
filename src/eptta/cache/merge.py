"""Streaming verified merge of independently produced feature shards."""
from eptta.cache.reader import FeatureCache
from eptta.cache.writer import FeatureCacheWriter
from eptta.errors import DataError


def merge_feature_caches(shard_refs, output, expected_ids):
    shards = [FeatureCache(ref) for ref in shard_refs]
    if not shards:
        raise DataError("no cache shards to merge")
    identity = shards[0].index["identity"]
    if any(shard.index["identity"] != identity for shard in shards[1:]):
        raise DataError("cache shards have different scientific/numerical/resource identities")
    from eptta.cache.keys import CacheIdentity
    cache_identity = CacheIdentity(**identity)
    with FeatureCacheWriter(output, cache_identity, expected_ids, shards[0].index["num_views"],
                            shards[0].index["feature_dim"]) as writer:
        for shard in shards:
            if shard.index["num_views"] != shards[0].index["num_views"] or shard.index[
                    "feature_dim"] != shards[0].index["feature_dim"]:
                raise DataError("cache shard shapes disagree")
            for ids, array in shard.iter_chunks():
                writer.add(ids, array)
    return FeatureCache(output).index
