from collections import defaultdict
from typing import Any, Dict, List, Tuple

from sortedcontainers import SortedKeyList

from galahad.server.dataclasses import Annotation


class Annotations:
    def __init__(self, text: str):
        self._text = text
        self._index: Dict[str, SortedKeyList] = defaultdict(lambda: SortedKeyList(key=_sort_func))

    @staticmethod
    def from_dict(text: str, annotations: Dict[str, List[Annotation]]) -> "Annotations":
        result = Annotations(text)

        for type_name, annotations_for_type in annotations.items():
            result._index[type_name].update(annotations_for_type)

        return result

    def to_dict(self) -> Dict[str, List[Annotation]]:
        result = defaultdict(list)
        for type_name, annotations in self._index.items():
            for annotation in annotations:
                result[type_name].append(
                    {"begin": annotation.begin, "end": annotation.end, "features": annotation.features}
                )

        return result

    def create_annotation(self, type_name, begin: int, end: int, features: Dict[str, Any] = None) -> Annotation:
        if features is None:
            features = {}

        annotation = Annotation(begin=begin, end=end, features=features)
        self._index[type_name].add(annotation)
        return annotation

    def get_covered_text(self, annotation: Annotation) -> str:
        return self._text[annotation.begin : annotation.end]

    def select(self, type_name: str) -> List[Annotation]:
        return list(self._index[type_name])

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

    @property
    def text(self) -> str:
        return self._text


def _sort_func(a: Annotation) -> Tuple[int, int]:
    return a.begin, a.end
