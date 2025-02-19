from copy import deepcopy
from datetime import timedelta, date
import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import (
    ClassifierMixin,
    OneToOneFeatureMixin,
    TransformerMixin,
    BaseEstimator,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from kma import zip_to_kma


class ConstantImputer(TransformerMixin, OneToOneFeatureMixin):
    """Impute missing values with a constant value.

    If the column is non-numeric, it's automatically cast to a category
    and added a missing category.
    """

    MISSING_CAT = "missing"
    MISSING_NUM = 0

    def __init__(self, constant=None):
        self.feature_names_in_ = None

    def fit(self, X, y=None):

        self.fill = pd.Series(
            [
                self.MISSING_CAT if X[c].dtype == "category" else self.MISSING_NUM
                for c in X
            ],
            index=X.columns,
        )

        return self

    def transform(self, X, y=None):
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        categoricals = X.select_dtypes(exclude=numerics).columns.to_list()

        for cat in categoricals:
            X[cat] = X[cat].astype("category")
            X[cat] = X[cat].cat.add_categories(self.MISSING_CAT)

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        return X.fillna(self.fill)

    def get_feature_names_out(self, names=None):
        return self.feature_names_in_


class IdTransformer(TransformerMixin):
    def __init__(self, hyperparam=None):
        self.hyperparam = hyperparam

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, names=None):
        return names

    def get_params(self, deep=True):
        return {"hyperparam": self.hyperparam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class MetricClassifier(BaseEstimator):
    """A Estimator that chooses between two models, a heuristic based, aka "Metric", one and a ML one.

    This model will give priority to the past data to decide if the carrier will reject or not,
    if there is no past data, it will use the ML model.
    """

    def __init__(
        self,
        metric_col: str,
        ml_model: BaseEstimator = DummyClassifier(
            strategy="stratified", random_state=42
        ),
        *args,
        **kwags,
    ) -> None:
        super().__init__(*args, **kwags)
        self.ml_model = ml_model
        self.metric_col = metric_col
        self.model_kwargs = kwags
        self.model_args = args
        self.suffix = "pipeline-3__"
        self.metric = None

    def fit(self, X, y, **fit_params):
        """Fits the ml model and fetches the past tender rejection metric"""
        s = self.suffix
        metric_col = s + self.metric_col
        epoch_col = s + "tender_sent_at_epoch"
        carrier_id_col = s + "rate_carrier_id"
        self.to_drop = [epoch_col, carrier_id_col, metric_col]
        self.metric = (
            X.loc[X[metric_col].notna(), :]
            .sort_values(by=epoch_col)[[carrier_id_col, metric_col]]
            .drop_duplicates(subset=carrier_id_col, keep="last")
            .reset_index(drop=True)
            .copy(deep=True)
        )
        self.ml_model = self.ml_model.fit(X.drop(self.to_drop, axis=1), y, **fit_params)
        return self

    def _predict(self, row, predict_proba=False):
        """Predicts based on the past metric or the ML model if there is no past metric.

        If predict_proba is true calculate a 2xN series with the probability of the
        carrier accepting or rejecting the tender.
        """
        s = self.suffix

        past_metric = self.metric.loc[
            self.metric[s + "rate_carrier_id"] == row[s + "rate_carrier_id"],
            s + self.metric_col,
        ]
        if past_metric.empty:
            x = row.drop(self.to_drop).values.reshape(1, -1)
            if predict_proba:
                return pd.Series(
                    self.ml_model.predict_proba(x)[0], index=[0, 1], dtype=float
                )
            return self.ml_model.predict(x).astype(float)[0]
        if predict_proba:
            return pd.Series(
                [1 - past_metric.values[0], past_metric.values[0]],
                index=[0, 1],
                dtype=float,
            )
        return past_metric.astype(float).values[0]

    def predict(self, X):
        return X.apply(self._predict, axis=1).values

    def predict_proba(self, X):
        result = X.apply(self._predict, axis=1, predict_proba=True)
        return result.values

    def get_params(self, deep=True):
        return {
            "ml_model": self.ml_model,
            "metric_name": self.metric_col,
            "model_kwargs": self.model_kwargs,
            "model_args": self.model_args,
            "suffix": self.suffix,
            "metric": self.metric,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def time_series_validation(
    X: pd.DataFrame,
    y: pd.DataFrame,
    base_data: pd.DataFrame,
    model: ClassifierMixin,
    n_folds=10,
    n_days_test=7,
    date_col="sqr_quote_created",
) -> pd.DataFrame:
    """Time Series Validation for panel data.

    This will split the data into a train set in the past and a test set in the future.
    """
    quote_day = base_data[date_col]
    last_day = base_data[date_col].max()
    splits = []

    for _ in range(n_folds):
        last_day -= timedelta(days=n_days_test)
        splits.append(last_day)

    splits.reverse()

    metrics = []

    for fold, split in enumerate(splits):
        # copying to avoid overwriting the original pipe
        model = deepcopy(model)

        print(f"Executing fold {fold}")

        train_X = X[quote_day <= split]
        test_X = X[
            (quote_day > split) & (quote_day <= split + timedelta(days=n_days_test))
        ]
        train_y = y[quote_day <= split]
        test_y = y[
            (quote_day > split) & (quote_day <= split + timedelta(days=n_days_test))
        ]

        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        y_pred_train = model.predict(train_X)

        y_score = model.predict_proba(test_X)[::, 1]
        y_score_train = model.predict_proba(train_X)[::, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_true=test_y, y_pred=y_pred, normalize="all"
        ).ravel()

        metric = pd.DataFrame(
            {
                "fold": [fold],
                "n_train": len(train_y),
                "n_test": len(test_y),
                "f1_train": [f1_score(y_pred=y_pred_train, y_true=train_y)],
                "recall_train": [recall_score(y_pred=y_pred_train, y_true=train_y)],
                "precision_train": [
                    precision_score(y_pred=y_pred_train, y_true=train_y)
                ],
                "roc_auc_train": [roc_auc_score(y_score=y_score_train, y_true=train_y)],
                "accuracy_train": [accuracy_score(y_pred=y_pred_train, y_true=train_y)],
                "f1_validation": [f1_score(y_pred=y_pred, y_true=test_y)],
                "recall_validation": [recall_score(y_pred=y_pred, y_true=test_y)],
                "precision_validation": [precision_score(y_pred=y_pred, y_true=test_y)],
                "roc_auc_validation": [roc_auc_score(y_score=y_score, y_true=test_y)],
                "accuracy_validation": [accuracy_score(y_pred=y_pred, y_true=test_y)],
                "true_postive_validation": tp,
                "false_postive_validation": fp,
                "true_negative_validation": tn,
                "false_negative_validation": fn,
            },
            index=[fold],
        )
        metrics.append(metric)

    results = pd.concat(metrics, axis=0)
    return results


def plot_boxplots(
    data, mask, by="is_rejection", description="if the tender rejected or accept"
):
    data = data[mask].copy()
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))

    fig.suptitle(f"Boxplots by '{description}', without outliers")

    data.plot.box(by=by, column="total_rates_provided", showfliers=False, ax=ax[0, 0])

    ax[0, 0].set_ylabel("Total Rates Provided")
    ax[0, 0].set_xlabel(by.replace("_", " ").capitalize())
    ax[0, 0].set_title("Total Rates Provided")

    data.plot.box(by=by, column="rate_pct_pos", showfliers=False, ax=ax[1, 0])

    ax[1, 0].set_ylabel(
        "Proportional Bid Rank in % \n(0% is the first bid and 100% is the last)"
    )
    ax[1, 0].set_xlabel(by.replace("_", " ").capitalize())
    ax[1, 0].set_title("Proportional Bid Rank in %")

    data[~data["is_loadsmart"]].plot.box(
        by=by, column="rate_over_ls", showfliers=False, ax=ax[0, 1]
    )

    ax[0, 1].set_ylabel("Flat Rate over LS Rate")
    ax[0, 1].set_xlabel(by.replace("_", " ").capitalize())
    ax[0, 1].set_title("Flat Rate over LS Rate, without Loadsmart")

    data.plot.box(by=by, column="rate_over_avg", showfliers=False, ax=ax[1, 1])

    ax[1, 1].set_ylabel("Flat Rate over Avg Rate")
    ax[1, 1].set_xlabel(by.replace("_", " ").capitalize())
    ax[1, 1].set_title("Flat Rate over Avg Rate")


def ztk(zipcode):
    try:
        return zip_to_kma(zipcode).id
    except:
        return None


def geo_features(data):
    data["origin_zipcode"] = data["origin_zipcode"].astype("string").str[:5]
    data["dest_zipcode"] = data["dest_zipcode"].astype("string").str[:5]
    data["origin_zip3"] = data["origin_zipcode"].astype("string").str[:3]
    data["dest_zip3"] = data["dest_zipcode"].astype("string").str[:3]
    data["origin_kma"] = data["origin_zipcode"].astype("string").str[:3].apply(ztk)
    data["dest_kma"] = data["dest_zipcode"].astype("string").str[:3].apply(ztk)
    data["origin_city"] = data["origin_city"].astype("string").str.lower()
    data["dest_city"] = data["dest_city"].astype("string").str.lower()
    data["origin_state"] = data["origin_state"].astype("string").str.upper()
    data["dest_state"] = data["dest_state"].astype("string").str.upper()
    data["origin_country"] = data["origin_country"].astype("string").str.upper()
    data["dest_country"] = data["dest_country"].astype("string").str.upper()
    return data


def set_epoch_na(x: float) -> float:
    try:
        return x if x > 0 else None
    except:
        return None


def date_features(data: pd.DataFrame) -> pd.DataFrame:
    data["days_en_route"] = (
        data["delivery_date"] - data["pickup_date"]
    ) / pd.to_timedelta(1, unit="D")

    data["del_b4_pu"] = data["days_en_route"] < 0

    data.loc[data["days_en_route"] < 0, "days_en_route"] = 0.0

    is_weekend_pickup = data["pickup_date"].dt.weekday >= 5
    is_weekend_delivery = data["delivery_date"].dt.weekday >= 5
    is_next_week_trip = (
        data["pickup_date"].dt.weekday > data["delivery_date"].dt.weekday
    ) | (data["days_en_route"] >= 7)
    is_weekend_trip = is_weekend_pickup | is_weekend_delivery | is_next_week_trip
    is_next_day_trip = (
        ((data["pickup_date"].dt.hour / 24) + data["days_en_route"])
    ) > 1
    delivery_date_epoch = data["pickup_date"].astype(int)

    data["is_weekend_pickup"] = is_weekend_pickup
    data["delivery_date_epoch"] = delivery_date_epoch
    data["pickup_date_epoch"] = data["pickup_date"].astype(int).apply(set_epoch_na)
    data["is_weekend_delivery"] = is_weekend_delivery
    data["is_next_day_trip"] = is_next_day_trip
    data["is_next_week_trip"] = is_next_week_trip
    data["is_weekend_trip"] = is_weekend_trip

    return data


def is_holiday(date: pd.DataFrame, us_holidays: dict) -> bool:
    if us_holidays.get(date):
        return True
    return False


def datetime_to_features(data: pd.DataFrame, column: str) -> pd.DataFrame:
    us_holidays = holidays.US(years=[2020, 2021, 2022, 2023])
    data = data.copy()
    data[f"{column}_year"] = data[column].dt.year
    data[f"{column}_month"] = data[column].dt.month
    data[f"{column}_quarter"] = data[column].dt.quarter
    data[f"{column}_week"] = data[column].dt.weekofyear
    data[f"{column}_day"] = data[column].dt.day
    data[f"{column}_hour"] = data[column].dt.hour
    data[f"{column}_dow"] = data[column].dt.day_of_week
    data[f"{column}_doy"] = data[column].dt.day_of_year
    data[f"{column}_is_holiday"] = (
        data[column].apply(lambda x: is_holiday(x, us_holidays)).astype("bool")
    )
    data[f"{column}_month_start"] = data[f"{column}_day"] <= 7
    data[f"{column}_month_end"] = data[f"{column}_day"] >= 23
    data[f"{column}_weekend"] = data[f"{column}_dow"].isin([5, 6])
    data[f"{column}_epoch"] = data[column].astype("int64")

    return data


def lead_time_features(data: pd.DataFrame, is_tender: bool = False) -> pd.DataFrame:
    data["lead_time_quote"] = (
        data["pickup_date"] - data["sqr_quote_created"]
    ) / pd.to_timedelta(1, unit="D")
    data["lead_time_rate"] = (
        data["pickup_date"] - data["rate_date"]
    ) / pd.to_timedelta(1, unit="D")
    data["quote_age"] = (
        data["rate_date"] - data["sqr_quote_created"]
    ) / pd.to_timedelta(1, unit="D")

    if is_tender:
        data["lead_time_tender"] = (
            data["pickup_date"] - data["tender_sent_at"]
        ) / pd.to_timedelta(1, unit="D")
        data["tender_age"] = (
            data["rate_date"] - data["tender_sent_at"]
        ) / pd.to_timedelta(1, unit="D")
    return data


def rate_features(data: pd.DataFrame) -> pd.DataFrame:
    rates_features = (
        data[["rate_id", "spot_quote_id", "flat_rate"]]
        .sort_values(["spot_quote_id", "rate_id"])
        .set_index("rate_id")
        .groupby(["spot_quote_id"])
        .expanding(min_periods=1)
        .agg({"flat_rate": ["mean", "std", "median", "min", "max", "count"]})
    )
    rates_features = (
        rates_features.reset_index().set_index("rate_id").drop("spot_quote_id", axis=1)
    )
    rates_features.columns = ["_".join(col) for col in rates_features.columns.values]
    data = data.set_index("rate_id").join(rates_features, how="left").reset_index()
    # removing duplicates that can appear with the join
    data = data.sort_values("flat_rate_mean", ascending=False).drop_duplicates(
        subset="rate_id", keep="first"
    )
    data["rate_above_mean"] = data["flat_rate"] > data["flat_rate_mean"]
    data["rate_above_median"] = data["flat_rate"] > data["flat_rate_median"]
    data["rate_is_currently_lowest"] = data["flat_rate"] == data["flat_rate_min"]
    data["rate_is_currently_high"] = data["flat_rate"] == data["flat_rate_max"]
    return data


def rejection_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["is_rejection"] = data["is_rejection"].astype(int)
    tender_features = (
        data[["rate_id", "rate_carrier_id", "is_rejection"]]
        .sort_values(["rate_carrier_id", "rate_id"])
        .set_index("rate_id")
        .groupby(["rate_carrier_id"])
        .rolling(
            window=10000, min_periods=1, closed="left"
        )  # adding a big number to basically be an expanding window that disconsiders the current row
        .agg({"is_rejection": ["mean", "std", "median", "sum", "count"]})
    )
    tender_features = (
        tender_features.reset_index()
        .set_index("rate_id")
        .drop("rate_carrier_id", axis=1)
    )
    tender_features.columns = ["_".join(col) for col in tender_features.columns.values]
    data = data.set_index("rate_id").join(tender_features, how="left").reset_index()
    # removing duplicates that can appear with the join
    data = data.sort_values("is_rejection_mean", ascending=False).drop_duplicates(
        subset="rate_id", keep="first"
    )
    data = data.rename({"is_rejection_count": "carrier_tender_count"}, axis=1)
    data["is_rejection_mean_15"] = data["is_rejection_mean"] > 0.15
    data["is_rejection_mean_50"] = data["is_rejection_mean"] > 0.5
    data["is_rejection_mean_70"] = data["is_rejection_mean"] > 0.7
    data["is_rejection_mean_90"] = data["is_rejection_mean"] > 0.9
    data["is_rejection_median_15"] = data["is_rejection_median"] > 0.15
    data["is_rejection_median_50"] = data["is_rejection_median"] > 0.5
    data["is_rejection_median_70"] = data["is_rejection_median"] > 0.7
    data["is_rejection_median_90"] = data["is_rejection_median"] > 0.9
    return data
