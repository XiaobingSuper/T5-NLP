import json
from typing import Dict, List

import numpy as np
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TransformerMixin
from sklearn.metrics import silhouette_score


def get_average_of_dict(dict: Dict[str, float]) -> float:
    if not dict:
        return 0

    total = 0.0
    num_elements = len(dict)
    for key in dict:
        total += dict[key]
    return total / num_elements


def write_scores_to_text_file(dict: Dict[str, float], file_path: str) -> None:
    serialized_scores = json.dumps(dict, indent=4)
    try:
        with open(file_path, "w+") as f:
            f.write(serialized_scores)
    except IOError as e:
        raise IOError(f"Error: {e}, could not open or write to the file: {file_path}")


def compute_features(
    feature_extractor: TransformerMixin, documents: List[str]
) -> np.ndarray:
    feature_matrix = feature_extractor.fit_transform(documents)
    return feature_matrix  # Out: [num_samples, num_features]


def cluster_from_features(
    feature_matrix: np.ndarray, clustering_algorithm: ClusterMixin
) -> np.ndarray:
    labels = clustering_algorithm.fit_predict(feature_matrix)
    return labels


# determines the number of clusters automatically accroding to the silhouette criterion
def kmeans_with_silhouette(documents: List[str], max_num_clusters) -> np.ndarray:
    feature_extractor = TfidfVectorizer(
        input="content",  # for passing strings
        lowercase=True,
        analyzer="word",  # we do not need a tokenizer/subwords.
        max_df=1.0,
        min_df=5,  # not to add very specific variable names that only occur in a single smaple
        max_features=15,  # instructions are very compact, 15 words should be enough
        binary=False,  # to distinguish between the number of occurences
        use_idf=True,  # known to work better than pure tf/bow
        smooth_idf=True,  # the default
    )
    feature_matrix = compute_features(feature_extractor, documents)

    num_cluster_candidates = np.arange(2, max_num_clusters + 1)
    maximum_silhouette_score = -1  # minimum value for silhouette score

    final_labels: np.ndarray
    for num_cluster in num_cluster_candidates:
        c_algorithm = KMeans(
            n_clusters=num_cluster,
            init="k-means++",  # k-means++ is known to work well and also the default
            n_init=10,  # default value, should be enough for our use cases
            max_iter=1000,  # increased it a bit since it does not increase complexity that much
            random_state=42,
            algorithm="auto",  # best to let sklearn to decide which algo to use
        )
        labels = cluster_from_features(
            feature_matrix=feature_matrix, clustering_algorithm=c_algorithm
        )
        silhouette_average = silhouette_score(X=feature_matrix, labels=labels)
        if silhouette_average > maximum_silhouette_score:
            maximum_silhouette_score = silhouette_average
            final_labels = labels

    return final_labels
