"""Model contracts and author architecture descriptions."""
from eptta.models.author import (ARCHITECTURES, canonical_to_native,
                                 class_weights_native, get_author_architecture,
                                 inspect_author_repository)

__all__ = ["ARCHITECTURES", "canonical_to_native", "class_weights_native",
           "get_author_architecture", "inspect_author_repository"]
