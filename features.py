import numpy as np
import pandas as pd

# raw columns the model takes as input (Revenue is the target)
INPUT_COLS = [
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
    "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]
NUM_INPUTS = INPUT_COLS[:10]
CAT_COLS = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
NEW_COLS = ["total_pages", "total_duration", "avg_product_time", "product_page_share",
            "has_page_value", "is_returning", "is_special_day"]


def add_features(df):
    df = df.copy()
    df["Weekend"] = df["Weekend"].astype(int)
    df["total_pages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["total_duration"] = (df["Administrative_Duration"] + df["Informational_Duration"]
                            + df["ProductRelated_Duration"])
    # sessions with 0 product pages would divide by zero, give them 0
    df["avg_product_time"] = (df["ProductRelated_Duration"] / df["ProductRelated"].replace(0, np.nan)).fillna(0)
    df["product_page_share"] = (df["ProductRelated"] / df["total_pages"].replace(0, np.nan)).fillna(0)
    df["has_page_value"] = (df["PageValues"] > 0).astype(int)
    df["is_returning"] = (df["VisitorType"] == "Returning_Visitor").astype(int)
    df["is_special_day"] = (df["SpecialDay"] > 0).astype(int)
    return df
