"""
Implements the base classes for different purposes. All child classes may
inherit from these base classes.
"""
from abc import ABCMeta, abstractmethod
from typing import Any

import pandas as pd


class BaseInference(metaclass=ABCMeta):
    @abstractmethod
    def infer(self):
        pass


class BaseTrainer(metaclass=ABCMeta):
    """
    A base model trainer class. All model trainers, e.g. neural network trainer,
    or ML model trainers like SVM, may inherit from this class. It only shares
    a train() method.
    """
    @abstractmethod
    def __init__(
            self,
            model: Any,
            *args: Any,
            **kwargs: Any
    ):
        """
        Initialises the class instance. This class cannot be initialised
        directly; only the child classes that inherit from this class can be
        initialised.

        Extra *args and **kwargs allow for different requirements from different
        trainers. For example, a neural network trainer may also require batch
        size, learning rate, or the device (cpu, cuda or mps), whereas an SVM
        trainer may not need these information.
        :param model: Any
            The model class. Can be any classes of models, for example
        :param X_train: pd.DataFrame
            The training data.
        :param y_train: pd.Series
            The training labels.
        :param X_val: pd.DataFrame
            The validation data.
        :param y_val: pd.Series
            The validation labels.
        :param args: Any
            Extra positional parameters for the trainer, to allow for different
            implementations.
        :param kwargs: Any
            Extra keyword parameters for the trainer, to allow for different
            implementations.
        """
        self.model = model

    @abstractmethod
    def train(
            self,
            train_data: Any,
            val_data: Any
    ) -> Any:
        """

        :param train_data: Any
            Object representing the training data. The object needs to have an
            internal representation of the data and the labels.
        :param val_data:
            Object representing the validation data. The object needs to have an
            internal representation of the data and the labels.
        :return: Any
        """
        pass


class BaseValidator(metaclass=ABCMeta):
    """
    A base model validator class. All model validators may inherit from this
    class. Abstract methods include __init__() and cross_validate().

    The general pipeline for cross validation is the following:
    1. Take X_train, y_train and feed into a StratifiedKFold object.
    2. Split the dataset into K folds of (X_train, y_train, X_val, y_val).
    3. Scale each split.
    4. Train and validate, using a specified metric.
    5. Take the best model from the split that has the best metric to be the
       final model.
    """
    @abstractmethod
    def __init__(
            self,
            n_splits,
            shuffle,
            random_state,
            *args,
            **kwargs
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    @abstractmethod
    def cross_validate(self, *args: Any, **kwargs: Any):
        pass


class BasePreprocessor(metaclass=ABCMeta):
    """
    A base preprocessor class. All data preprocessors may inherit from this
    class.

    The general pipeline for preprocessing a cleaned dataset in Pandas DataFrame
    format is follows:
    1. Separate the labels from the data. The label needs to be named 'label',
       otherwise there will be unexpected issues in parsing.
    2. Separate any other columns that are not part of the data itself, e.g.
       metadata such as the name of an image.
    3. If the data has not been train-test split, a split needs to be performed.
       This is not needed for data that has already been split.
    4. For each dataset (train/test), scale the data according to the mean/stdev
       of the training set. Then use that to transform the test set to avoid
       data leakage.
    5.

    After the get_data_and_labels() method is called on the object, the
    separated datasets are accessible via the attributes: X_train_, X_test_,
    y_train_, y_test_.

    After the

    Any other attributes are declared as needed in the child class.
    """
    @abstractmethod
    def __init__(self):
        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        self.scaler_ = None

    @abstractmethod
    def get_data_and_labels(
            self,
            df_train: pd.DataFrame,
            df_test: pd.DataFrame
    ):
        """
        Separates the data from the labels. Assumes the whole dataset has been
        train-test split.

        :param df_train: pd.DataFrame
            The training dataset.
        :param df_test: pd.DataFrame
            The testing dataset.

        :return:
        """
        pass

    @abstractmethod
    def scale(
            self,
            scaler: Any,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame
    ) -> Any:
        """
        Scales the data according to the scaler object. The general syntax to
        scale a dataset with the scaler object from the sklearn API is via the
        following:
        ```
        scaler = Scaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        ```

        After scaling, the scaler object should be stored for later use, from
        which the following attributes are available:
        ```
        scale_            : ndarray of shape (n_features,) or None
        mean_             : ndarray of shape (n_features,) or None
        var_              : ndarray of shape (n_features,) or None
        n_features_in_    : int
        feature_names_in_ : ndarray
        n_samples_seen_   : int
        ```

        :param scaler: Any
        :param X_train:
        :param X_test:
        :return:
        """
        pass


class BaseSaver(metaclass=ABCMeta):
    """
    Base interface for a saver class. It can be utilised for cross-validation
    model saving.
    """
    @abstractmethod
    def __init__(self, save_path):
        self.save_path = save_path


class BasePlotter(metaclass=ABCMeta):
    """
    Base interface for a plotter class. It can be utilised for plotting training
    and validation results.
    """
    @abstractmethod
    def __init__(self):
        pass


class BaseTester(metaclass=ABCMeta):
    """
    Base interface for a tester class. It takes the path to the trained model,
    results on unseen data, and plots test confusion matrix and test accuracy.
    """