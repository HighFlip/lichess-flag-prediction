import argparse
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import add_features, to_records

# best parameters found in section 4.4 of the notebook
XGB_PARAMS = {"eta": 0.05, "max_depth": 4, "min_child_weight": 50,
              "objective": "binary:logistic", "eval_metric": "auc",
              "nthread": 8, "seed": 1, "verbosity": 1}
NUM_BOOST_ROUND = 357


def load_data(path):
    df = pd.read_csv(path)
    df = df[df["opp_clock"] <= df["time_control"]] # opponent was given extra time (+15 s button)
    df = df[df["ply"] > 10] # under 60 s within the first 10 plies, most likely a disconnection
    return df.reset_index(drop=True)


def split(df):
    """Same group split by game_id as the notebook, returns (full_train, test)."""
    games_full_train, games_test = train_test_split(df["game_id"].unique(), test_size=0.2, random_state=1)
    df_full_train = df[df["game_id"].isin(games_full_train)].reset_index(drop=True)
    df_test = df[df["game_id"].isin(games_test)].reset_index(drop=True)
    return df_full_train, df_test


def train(df_full_train):
    df_full_train = add_features(df_full_train)

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(to_records(df_full_train))
    dtrain = xgb.DMatrix(X, label=df_full_train["flagged"].values, feature_names=list(dv.get_feature_names_out()))
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return dv, model


def predict(df, dv, model):
    X = dv.transform(to_records(add_features(df)))
    return model.predict(xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="data/snapshots.csv")
    parser.add_argument("-o", "--output", default="model.bin")
    args = parser.parse_args()

    df = load_data(args.data)
    df_full_train, df_test = split(df)

    dv, model = train(df_full_train)
    auc = roc_auc_score(df_test["flagged"], predict(df_test, dv, model))
    print(f"trained on {len(df_full_train)} rows, test AUC = {auc:.4f}")

    with open(args.output, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
