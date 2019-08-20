from typing import Set, Tuple, List, Dict

from allennlp.data import Token


class Cluster:
    def __init__(self):
        self.entities: Set[Tuple[str, Tuple[int, int]]] = set()

    def __repr__(self):
        return repr(self.entities)


class ClusterManager:

    def __init__(self,
                 flat_tokens: List[Token],
                 flat_text: str):
        self._locked = False
        self._clusters: List[Cluster] = []
        self._interactions: Dict[Tuple[int, int], str] = dict()

        self._flat_tokens = flat_tokens
        self._flat_text = flat_text

        self._name_registry: Dict[str, Cluster] = dict()
        self._span_registry: Dict[Tuple[int, int], Cluster] = dict()
        self._cluster_registry: Dict[int, Cluster] = dict()

    @property
    def clusters(self) -> List[Cluster]:
        return self._clusters

    @property
    def interactions(self) -> Dict[Tuple[int, int], str]:
        return self._interactions

    @property
    def raw_interactions(self) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]:
        return {
            (first_mention, second_mention): label
            for (first_cluster, second_cluster), label in self._interactions.items()
            for _, first_mention in self._clusters[first_cluster].entities
            for _, second_mention in self._clusters[second_cluster].entities
        }

    def merge_clusters(self, target: Cluster, source: Cluster):
        if target is None:
            return source
        if source is None:
            return target

        for name, span in source.entities:
            self._name_registry[name] = target
            self._span_registry[span] = target
            target.entities.add((name, span))

        return target

    def resolve_mention(self, first_idx: int, last_idx: int) -> Tuple[str, Tuple[int, int]]:
        first = self._flat_tokens[first_idx]
        last = self._flat_tokens[last_idx]

        start = first.idx
        end = last.idx + len(last.text)
        span = (start, end)

        name = self._flat_text[start:end]

        return name, span

    def add_mention(self,
                    first_idx: int,
                    last_idx: int,
                    cluster_id: int = None):

        if self._locked and cluster_id is not None:
            raise ValueError('Merging by cluster_id isn\'t possible after locking.')

        name, span = self.resolve_mention(first_idx, last_idx)

        new_mention = Cluster()
        new_mention.entities.add((name, span))

        by_span = self._span_registry.get(span)
        new_mention = self.merge_clusters(by_span, new_mention)

        by_name = self._name_registry.get(name)
        new_mention = self.merge_clusters(by_name, new_mention)

        by_cluster = self._cluster_registry.get(cluster_id)
        new_mention = self.merge_clusters(by_cluster, new_mention)

        self._name_registry[name] = new_mention
        self._span_registry[span] = new_mention
        if cluster_id is not None:
            self._cluster_registry[cluster_id] = new_mention

        if not self._locked:
            return None

        try:
            idx = self._clusters.index(new_mention)
        except ValueError:
            idx = len(self._clusters)
            self._clusters.append(new_mention)

        return idx

    def add_interaction(self,
                        participants: Tuple[Tuple[int, int], Tuple[int, int]],
                        label: str):
        first_span, second_span = participants
        first_cluster = self.add_mention(*first_span)
        second_cluster = self.add_mention(*second_span)

        self._interactions[first_cluster, second_cluster] = label

    # def add_cluster(self, cluster: List[Tuple[int, int]]):
    #     entity = {
    #         "is_state": False,
    #         "label": "TODO",  # TODO
    #         "is_mentioned": True,
    #         "is_mutant": False,
    #         "names": {}
    #     }
    #
    #     for first_idx, last_idx in cluster:
    #         first = self.flat_tokens[first_idx]
    #         last = self.flat_tokens[last_idx]
    #
    #         start = first.idx
    #         end = last.idx + len(last)
    #
    #         name = flat_text[start:end]

    def check(self):
        assert set(self._span_registry.values()) == set(self._name_registry.values())

    def lock_clusters(self):
        self._locked = True
        # enumerate ...
        self.check()

        clusters = set(self._span_registry.values())
        self._clusters = list(clusters)
