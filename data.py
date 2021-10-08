#!/usr/bin/env python3

import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)


class CountryInfo:
    """Essentially a struct for storing information about countries"""

    def __init__(self, row):
        self.country_name = row["country_name"]
        self.longitude = row["longitude"]
        self.latitude = row["latitude"]


def load_data():
    """Loads data from csv files

    returns:
        code_to_info - dictionary from code (string) to CountryInfo
        edges - set of country code pairs (set of sets)
    """
    df_loc = pd.read_csv("country-locations.csv", keep_default_na=False, na_values=[""])
    df_loc.dropna(axis=0, inplace=True)
    # print(df_loc.head())

    df_borders = pd.read_csv("country-borders.csv", keep_default_na=False, na_values=[""])
    # print(df_borders.head())

    A = set(df_loc["country_code"])
    B = set(df_borders["country_code"])
    # print("Mismatched elements:")
    # for country_code in A-B:
    #     print(" ", country_code, df_loc[df_loc["country_code"] == country_code]["country_name"])
    # for country_code in B-A:
    #     print(" ", country_code)
    country_codes = A & B

    # maps country_code to tuple (longitude, latitude)
    code_to_info = dict()
    # collection of border pais ie. ("US", "CA") where order of border pairs doesn't matter
    edges = set()
    for index, row in df_loc.iterrows():
        country_code = row["country_code"]
        if country_code in country_codes:
            code_to_info[country_code] = CountryInfo(row)

    for index, row in df_borders.dropna().iterrows():
        country_code_0 = row["country_code"]
        country_code_1 = row["country_border_code"]
        if country_code_0 in country_codes and country_code_1 in country_codes:
            edges.add(frozenset({country_code_0, country_code_1}))

    return code_to_info, edges


if __name__ == '__main__':
    load_data()
