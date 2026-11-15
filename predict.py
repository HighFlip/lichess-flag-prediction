import pickle

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

from features import add_features, to_records

with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="lichess-flag-prediction")


class Snapshot(BaseModel):
    """One row of data/snapshots.csv, without game_id and flagged."""
    threshold: int
    event: str
    time_control: int
    ply: int
    is_white: bool
    my_clock: float
    opp_clock: float
    my_rating: int
    opp_rating: int
    my_material: int
    opp_material: int
    my_berserk: bool
    opp_berserk: bool
    my_avg_move_time: float
    opp_avg_move_time: float
    n_pieces: int
    queens_on: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(snapshot: Snapshot):
    df = add_features(pd.DataFrame([snapshot.model_dump()]))
    X = dv.transform(to_records(df))
    dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    probability = float(model.predict(dmatrix)[0])
    return {"flag_probability": round(probability, 4), "will_flag": probability >= 0.5}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9696)
