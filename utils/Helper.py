"""Defines various helper functions to create EventStore object from given source."""

import os
import csv
from urllib.request import urlopen

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    requests = None
from datamodel.Event import IntervalEvent


class _SimpleSeries:
    def __init__(self, values):
        self._values = values
        self.iloc = self

    def __getitem__(self, idx):
        return self._values[idx]

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)


class _SimpleDataFrame:
    def __init__(self, columns, rows):
        self.columns = columns
        self._rows = rows

    def iterrows(self):
        for idx, row in enumerate(self._rows):
            yield idx, _SimpleSeries(row)


def _coerce_cell(value: str):
    value = value.strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_csv_fallback(path: str, sep: str, header):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        rows = list(reader)

    if not rows:
        return _SimpleDataFrame([], [])

    if not header:
        columns = [c.strip() for c in rows[0]]
        data_rows = rows[1:]
    else:
        columns = list(header)
        data_rows = rows

    coerced_rows = [[_coerce_cell(cell) for cell in row] for row in data_rows]
    return _SimpleDataFrame(columns, coerced_rows)


def getDataframe(src, local=False, sep="\t", header=None):
    """Helper function to return a data frame
    Local is boolean, if local then source should be path to the file
    Otherwise it should be a URL to the the file
    """

    if pd is None:
        if not local:
            if "dropbox" in src:
                src = src.replace("dl=0", "dl=1")
            if requests is not None:
                req = requests.get(src)
                url_content = req.content
            else:
                url_content = urlopen(src).read()
            with open("data.txt", "wb") as csv_file:
                csv_file.write(url_content)
            try:
                data_frame = _read_csv_fallback("data.txt", sep, header)
            finally:
                os.remove("data.txt")
            return data_frame
        return _read_csv_fallback(src, sep, header)

    if not local:
        # To force a dropbox link to download change the dl=0 to 1
        if "dropbox" in src:
            src = src.replace("dl=0", "dl=1")
        # Download the CSV at url
        req = requests.get(src)
        urlContent = req.content
        csvFile = open("data.txt", "wb")
        csvFile.write(urlContent)
        csvFile.close()
        # Read the CSV into pandas
        # If header list is empty, the dataset provides header so ignore param
        if header is None:
            dataFrame = pd.read_csv("data.txt", sep)
        # else use header param for column names
        else:
            dataFrame = pd.read_csv("data.txt", sep, names=header)
        # Delete the csv file
        os.remove("data.txt")
        # return dataFrame
    # Dataset is local
    else:
        # If header list is empty, the dataset provides header so ignore param
        if not header:
            dataFrame = pd.read_csv(src, sep=sep)
        # else use header param for column names
        else:
            dataFrame = pd.read_csv(src, sep=sep, names=header)
    return dataFrame


def getTimeToSortBy(evt):
    """Helper function for generateSequence to use when sorting events to get
    what time field to sort by. Also used in splitSequences to give the time of
    an event when splitting the events up
    """

    # Sort by starting time of event if its an interval event
    if isinstance(evt, IntervalEvent):
        return evt.time[0]
    # Otherwise use the timestamp
    return evt.timestamp


def insertEventIntoDict(key, dictionary, event):
    """Helper to insert an event into a map Params are key=unique id for that time,
    map of key to event list, event object
    """
    if key in dictionary:
        dictionary[key].append(event)
    else:
        dictionary[key] = [event]
