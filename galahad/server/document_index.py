from collections import defaultdict
from typing import Dict, List, Tuple

from sortedcontainers import SortedKeyList

from galahad.server.dataclasses import Annotation


class DocumentIndex:
    def __init__(self, annotations: Dict[str, List[Annotation]]):
        self._index: Dict[str, SortedKeyList] = defaultdict(lambda: SortedKeyList(key=_sort_func))

        for type_name, annotations_for_type in annotations.items():
            self._index[type_name].update(annotations_for_type)

    def select_covered(self, type_name: str, covering_annotation: Annotation) -> List[Annotation]:
        """Returns a list of covered annotations.

        Return all annotations that are covered
        Only returns annotations that are fully covered, overlapping annotations
        are ignored.

        Args:
            type_name: The type name of the annotations to be returned.
            covering_annotation: The name of the annotation which covers.

        Returns:
            A list of covered annotations
        """
        c_begin = covering_annotation.begin
        c_end = covering_annotation.end

        result = []
        for annotation in self._get_feature_structures_in_range(type_name, c_begin, c_end):
            if annotation.begin >= c_begin and annotation.end <= c_end:
                result.append(annotation)
        return result

    def _get_feature_structures_in_range(self, type_name: str, begin: int, end: int) -> List[Annotation]:
        """Returns a list of all feature structures of type `type_name`.

        Only features are returned that are in [begin, end] or close to it. If you use this function,
        you should always check bound in the calling method.
        """

        annotations = self._index[type_name]

        # We use binary search to find indices for the first and last annotations that are inside
        # the window of [begin, end].
        idx_begin = annotations.bisect_key_left((begin, begin))
        idx_end = annotations.bisect_key_right((end, end))

        return annotations[idx_begin:idx_end]


def _sort_func(a: Annotation) -> Tuple[int, int]:
    return a.begin, a.end
