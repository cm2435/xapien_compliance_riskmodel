from datetime import datetime
from typing import List

import numpy as np
from matplotlib.dates import date2num
from sklearn.cluster import DBSCAN


class TemporalModel:
    """
    A class for temporal modeling using DBSCAN clustering.

    Attributes:
        fitted_intervals (dict): Dictionary containing the fitted intervals for each cluster label.
    """

    def __init__(self):
        self.fitted_intervals: dict = None

    def fit(
        self, dates: List[dict], max_time_interval: float = 182.5, min_samples: int = 10
    ) -> dict:
        """
        Fit a clustering model to a series of dates to try to detect 'bursts' of news activity

        Args:
            dates (list): List of date dictionaries.
            max_time_interval (float): Maximum time interval for clustering in days. Default is 182.5 (half of 365).
            min_samples (int): Minimum number of samples required to form a cluster. Default is 10.

        Returns:
            dict: Dictionary containing the fitted intervals for each cluster label.
        """

        # Convert data to easier datetime format
        converted_dates = self._convert_dates(dates)

        # Perform temporal clustering using DBSCAN
        clustering = DBSCAN(eps=max_time_interval, min_samples=min_samples).fit(
            date2num(converted_dates).reshape(-1, 1)
        )
        labels = clustering.labels_
        unique_labels = set(labels)

        timeseries_bounds: dict = {}
        for label in unique_labels:
            if label == -1:  # Skip outliers (label -1)
                continue

            cluster_dates = np.array(converted_dates)[np.where(labels == label)]
            if len(cluster_dates) > min_samples:
                min_date = np.min(cluster_dates)
                max_date = np.max(cluster_dates)
                timeseries_bounds[label] = (min_date, max_date)

        return timeseries_bounds

    def predict(self, dates: List[dict]) -> List[int]:
        """
        Predict cluster labels for a series of dates in dictionary form

        Args:
            dates (list): List of date dictionaries.

        Returns:
            list: List of cluster labels corresponding to each date.
        """

        if self.fitted_intervals is None:
            self.fitted_intervals = self.fit(dates)

        date_labels: List[int] = []
        for date in dates:
            if date is None:
                date_labels.append(
                    -2
                )  # Not a news cycle label, but not -1 as this would be for a date that is present but is normal news
            else:
                cluster_label = self._get_cluster_label(date)
                date_labels.append(cluster_label)

        return date_labels

    def _get_cluster_label(self, date: dict) -> int:
        """
        Get the cluster label for a given date.

        Args:
            date (dict): A date in dictionary format with keys for year, month and day.

        Returns:
            int: Cluster label for the date.
        """

        cluster_label = -1
        for key, interval in self.fitted_intervals.items():
            if self._is_date_within_range(date, interval[0], interval[1]):
                cluster_label = key
        return cluster_label

    @staticmethod
    def _is_date_within_range(
        date: dict, min_date: datetime, max_date: datetime
    ) -> bool:
        """
        Check if a given date falls between a specified max and min

        Args:
            date (dict): Date dictionary.
            min_date (datetime): Minimum date of the range.
            max_date (datetime): Maximum date of the range.

        Returns:
            bool: True if the given date is within the range, False otherwise.
        """

        date_str = f"{date['Year']}-{date['Month']}-{date['Day']}"
        converted_date = datetime.strptime(date_str, "%Y-%m-%d")

        return min_date <= converted_date <= max_date

    @staticmethod
    def _convert_dates(dates: List[dict]) -> List[datetime]:
        """
        Convert the date dictionaries to datetime objects.

        Args:
            dates (list): List of date dictionaries.

        Returns:
            list: List of converted datetime objects.
        """

        converted_dates: List[datetime] = []
        for date in dates:
            if date is not None:
                date_str = f"{date['Year']}-{date['Month']}-{date['Day']}"
                converted_date = datetime.strptime(date_str, "%Y-%m-%d")
                converted_dates.append(converted_date)
        return converted_dates
