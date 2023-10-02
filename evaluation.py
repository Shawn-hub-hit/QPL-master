
import abc
import numpy as np
import math

class BaseMetric(abc.ABC):
    """Metric base class."""

    ALIAS = 'base_metric'

    @abc.abstractmethod
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Call to compute the metric.

        :param y_true: An array of groud truth labels.
        :param y_pred: An array of predicted values.
        :return: Evaluation of the metric.
        """

    @abc.abstractmethod
    def __repr__(self):
        """:return: Formated string representation of the metric."""

    def __eq__(self, other):
        """:return: `True` if two metrics are equal, `False` otherwise."""
        return (type(self) is type(other)) and (vars(self) == vars(other))

    def __hash__(self):
        """:return: Hashing value using the metric as `str`."""
        return str(self).__hash__()


class RankingMetric(BaseMetric):
    """Ranking metric base class."""

    ALIAS = 'ranking_metric'

def sort_and_couple(labels: np.array, scores: np.array) -> np.array:
    """Zip the `labels` with `scores` into a single list."""
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))


class DiscountedCumulativeGain(RankingMetric):
    """Disconunted cumulative gain metric."""

    ALIAS = ['discounted_cumulative_gain', 'dcg']

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`DiscountedCumulativeGain` constructor.
        :param k: Number of results to consider.
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate discounted cumulative gain (dcg).
        Relevance is positive real values or binary values.
        Example:
            >>> y_true = [0, 1, 2, 0]
            >>> y_pred = [0.4, 0.2, 0.5, 0.7]
            >>> DiscountedCumulativeGain(1)(y_true, y_pred)
            0.0
            >>> round(DiscountedCumulativeGain(k=-1)(y_true, y_pred), 2)
            0.0
            >>> round(DiscountedCumulativeGain(k=2)(y_true, y_pred), 2)
            2.73
            >>> round(DiscountedCumulativeGain(k=3)(y_true, y_pred), 2)
            2.73
            >>> type(DiscountedCumulativeGain(k=1)(y_true, y_pred))
            <class 'float'>
        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Discounted cumulative gain.
        """
        if self._k <= 0:
            return 0.
        coupled_pair = sort_and_couple(y_true, y_pred)
        result = 0.
        for i, (label, score) in enumerate(coupled_pair):
            if i >= self._k:
                break
            if label > self._threshold:
                result += (math.pow(2., label) - 1.) / math.log(2. + i)
        return result


class NormalizedDiscountedCumulativeGain(RankingMetric):
    """Normalized discounted cumulative gain metric."""

    ALIAS = ['normalized_discounted_cumulative_gain', 'ndcg']

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`NormalizedDiscountedCumulativeGain` constructor.
        :param k: Number of results to consider
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate normalized discounted cumulative gain (ndcg).
        Relevance is positive real values or binary values.
        Example:
            >>> y_true = [0, 1, 2, 0]
            >>> y_pred = [0.4, 0.2, 0.5, 0.7]
            >>> ndcg = NormalizedDiscountedCumulativeGain
            >>> ndcg(k=1)(y_true, y_pred)
            0.0
            >>> round(ndcg(k=2)(y_true, y_pred), 2)
            0.52
            >>> round(ndcg(k=3)(y_true, y_pred), 2)
            0.52
            >>> type(ndcg()(y_true, y_pred))
            <class 'float'>
        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Normalized discounted cumulative gain.
        """
        dcg_metric = DiscountedCumulativeGain(k=self._k,
                                              threshold=self._threshold)
        idcg_val = dcg_metric(y_true, y_true)
        dcg_val = dcg_metric(y_true, y_pred)
        return dcg_val / idcg_val if idcg_val != 0 else 0



class MeanReciprocalRank(RankingMetric):
    """Mean reciprocal rank metric."""

    ALIAS = ['mean_reciprocal_rank', 'mrr']

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`MeanReciprocalRankMetric`.
        :param k: Number of results to consider.
        :param threshold: The label threshold of relevance degree.
        """
        if k <= 0:
            raise ValueError(f"k must be greater than 0. {k} received.")
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate reciprocal of the rank of the first relevant item.
        Example:
            >>> import numpy as np
            >>> y_pred = np.asarray([0.2, 0.3, 0.7, 1.0])
            >>> y_true = np.asarray([1, 0, 0, 0])
            >>> MeanReciprocalRank(k=4)(y_true, y_pred)
            0.25
        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Mean reciprocal rank @ k.
        """
        coupled_pair = sort_and_couple(y_true, y_pred)
        for idx, (label, pred) in enumerate(coupled_pair):
            if idx >= self._k:
                break
            if label > self._threshold:
                return 1. / (idx + 1)
        return 0.


class Recall(RankingMetric):
    """Recall metric."""

    ALIAS = 'recall'

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`RecallMetric` constructor.
        :param k: Number of results to consider.
        :param threshold: the label threshold of relevance degree.
        """
        if k <= 0:
            raise ValueError(f"k must be greater than 0. {k} received.")
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate recall@k.
        Example:
            >>> y_true = [0, 0, 0, 1]
            >>> y_pred = [0.2, 0.4, 0.3, 0.1]
            >>> Recall(k=1)(y_true, y_pred)
            0.0
            >>> Recall(k=2)(y_true, y_pred)
            0.0
            >>> Recall(k=4)(y_true, y_pred)
            1.0
            >>> Recall(k=5)(y_true, y_pred)
            1.0
        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Recall @ k.
        """
        result = 0.
        pos = 0
        coupled_pair = sort_and_couple(y_true, y_pred)
        for idx, (label, score) in enumerate(coupled_pair):
            if label > self._threshold:
                pos += 1.
                if idx < self._k:
                    result += 1.
        if pos == 0:
            return 0.
        else:
            return result / pos