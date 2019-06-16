import logging

import pandas as pd


class TickerDataFramePreparer(object):
    """
    The DataFrame to be processed should contain at least the following columns:
    ['volume', 'open', close', 'high', 'low']
    with DateTime being the index column. The tickers need not be consecutive
    (no disjoints). If specified, Any disjointed data points will be overwritten
    by the latest data points before the missing data points.
    """

    def __init__(self, window="60s", window_start=None, dense=True):
        self.window = window
        self.window_start = window_start
        self.dense = dense
        self.logger = logging.getLogger(self.__class__.__name__)

    def average_columns(self, df, cols):
        return df[list(cols)].mean(axis=1)

    def squash(self, df, meancol_format="{}-v", window="60s", window_start=None,
               weighted_mean=("high", "low", "open", "close")):
        """
        Squash tickers into larger tickers
        :param df: tickers DataFrame
        :param window (str or timedelta): time interval to group data points into
        :param window_start: the starting point of the non-overlapping windows
                             defaults to the earliest curtime in the dataset
        :param offset: the starting index
        :param weighted_mean:
            list of columns to compute volume-weighted mean
            computed column will be have "-v" suffix (customizable)
        :return: squashed tickers DataFrame
        """
        if isinstance(window, str):
            # window = TimedeltaParser()(window)
            window = pd.to_timedelta(window)
        if window_start is None:
            window_start = df.index.min()

        # non-overlapping rolling
        # * df.rolling may not be suitable for stock data
        # groupby function maps each data point to the nearest
        # subsequent boundary

        def groupby(idx):
            return idx + (window - ((idx - window_start) % window))

        def rolling(df):
            return df.groupby(groupby)

        # col -> tuple( for-each-group function, pd.DataFrame.fillna kwargs )
        meta = {
            "high": (lambda r: r.max(), dict(
                method="pad"
            )),
            "low": (lambda r: r.min(), dict(
                method="pad"
            )),
            "volume": (lambda r: r.sum(), dict(
                value=1e-7
            )),
            "open": (lambda r: r.apply(lambda rows: rows[0]), dict(
                method="pad"
            )),
            "close": (lambda r: r.apply(lambda rows: rows[-1]), dict(
                method="pad"
            )),
        }
        ret = pd.DataFrame()
        for col, (apply_fn, fillna_kwargs) in meta.items():
            self.logger.info(f"aggregating '{col}'...")
            dtype = df[col].dtype
            feat = apply_fn(rolling(df[col]))

            if self.dense:
                dt_range = (feat.index.min(), feat.index.max())
                feat = feat.reindex(
                    pd.date_range(*dt_range, freq=window),
                    copy=False
                )
                feat.fillna(**fillna_kwargs, inplace=True)

            ret[col] = feat.astype(dtype, copy=False)

        ret["olhc"] = self.average_columns(ret,
                                           ("open", "low", "high", "close"))
        ret["lhc"] = self.average_columns(ret, ("low", "high", "close"))

        for col in weighted_mean:
            self.logger.info(f"aggregating and calculating weighted mean "
                             f"for {col}...")
            dtype = df[col].dtype
            value = df[col] * df["volume"]
            col_name = meancol_format.format(col)
            feat = rolling(value).sum() / ret["volume"]

            if self.dense:
                dt_range = (feat.index.min(), feat.index.max())
                feat = feat.reindex(
                    pd.date_range(*dt_range, freq=window),
                    copy=False
                )
                feat.fillna(method="pad", inplace=True)

            ret[col_name] = feat.astype(dtype, copy=False)

        return ret

    def prepare(self, df: pd.DataFrame):
        dtypes = {c: df[c].dtype for c in df.columns}

        # add small epsilon to prevent division-by-zero errors
        df["volume"] += 1e-7
        df["olhc"] = self.average_columns(df, ("open", "low", "high", "close"))
        df["lhc"] = self.average_columns(df, ("low", "high", "close"))

        df = self.squash(
            df=df,
            window=self.window,
            window_start=self.window_start,
            meancol_format="{}v",
            weighted_mean=("high", "low", "open", "close", "olhc", "lhc")
        )

        for c, dtype in dtypes.items():
            df[c] = df[c].astype(dtype, copy=False)

        return df
