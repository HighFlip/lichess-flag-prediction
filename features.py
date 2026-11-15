"""Feature engineering shared by train.py and predict.py (same steps as section 1 of the notebook)."""

import pandas as pd

NUM_COLS = ["threshold", "ply", "my_clock", "opp_clock", "my_rating", "opp_rating",
            "my_material", "opp_material", "clock_diff", "clock_ratio", "my_clock_frac",
            "opp_clock_frac", "rating_diff", "material_diff", "my_avg_move_time",
            "opp_avg_move_time", "n_pieces"]
CAT_COLS = ["time_control", "is_white", "my_berserk", "opp_berserk", "is_arena", "queens_on"]


def add_features(df):
    """Add the engineered columns to a dataframe of raw snapshots (columns of prepare_data.py)."""
    df = df.copy()
    df["is_arena"] = df["event"].str.contains("tournament") # "tournament" = arena, swiss events say "swiss"
    df.loc[~df["is_arena"], ["my_berserk", "opp_berserk"]] = False # berserk only exists in arenas

    df["clock_diff"] = df["my_clock"] - df["opp_clock"]
    df["clock_ratio"] = (df["my_clock"] + 1) / (df["opp_clock"] + 1) # avoids dividing by 0
    df["my_clock_frac"] = df["my_clock"] / df["time_control"]
    df["opp_clock_frac"] = df["opp_clock"] / df["time_control"]
    df["rating_diff"] = df["my_rating"] - df["opp_rating"]
    df["material_diff"] = df["my_material"] - df["opp_material"]

    df["time_control"] = df["time_control"].astype(str) # categorical
    return df


def to_records(df):
    return df[CAT_COLS + NUM_COLS].to_dict(orient="records")
