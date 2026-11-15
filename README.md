# Will They Flag? Lichess Blitz Time-Forfeit Prediction

ML Zoomcamp 2026 midterm project.

## Problem

In blitz chess, a lot of games are lost on time ("flagging") and not on the board. Once a player gets low on time, will they actually flag, or will they manage to finish the game?

This project takes a snapshot of a game the first time a player's clock drops below 60, 40, 20 and 10 seconds, and predicts the probability that this player goes on to lose on time. It could be used as a "time trouble" warning, a streamer overlay, or to review your own games and see in which positions you were most at risk.

The metric is ROC AUC. What matters is ranking the risky situations above the safe ones, and the classes are a bit imbalanced (39% of snapshots end in a flag).

## Dataset

Source is the [Lichess open database](https://database.lichess.org/), rated standard games from September 2026. `prepare_data.py` streams the first 200,000 games of the monthly dump and keeps the games played at 3+0, 5+0 and 10+0 that ended normally or on time. It then replays each game and records a snapshot every time a player crosses one of the thresholds, so a game gives up to 8 rows (4 thresholds × 2 players).

The result is committed as `data/snapshots.csv` (160,718 rows). To rebuild it (about 6 minutes):

```bash
uv run python prepare_data.py --games 200000
```

Columns ("my" is the player who just crossed the threshold, "opp" is their opponent):

| Column | Description |
|---|---|
| `game_id` | Lichess game id, only used for the split |
| `threshold` | The threshold crossed (60, 40, 20 or 10 s) |
| `event` | Lichess event name (rated game, arena or swiss tournament) |
| `eco`, `opening` | Opening code and name (dropped, they didn't help the model) |
| `time_control` | Base time in seconds (180, 300 or 600, no increment) |
| `ply` | Half-move number at the snapshot |
| `is_white` | Whether the player has the white pieces |
| `my_clock`, `opp_clock` | Seconds left on each clock |
| `my_rating`, `opp_rating` | Lichess ratings |
| `my_material`, `opp_material` | Material on the board (P=1, N=B=3, R=5, Q=9) |
| `my_berserk`, `opp_berserk` | Whether the player berserked (halved their clock in an arena), detected from the first clock |
| `my_avg_move_time`, `opp_avg_move_time` | Average seconds per move over the player's last 5 moves |
| `n_pieces` | Knights, bishops, rooks and queens left on the board |
| `queens_on` | Whether any queen is still on the board |
| `flagged` | Target, 1 if the player lost on time |

In the notebook we also add `clock_diff`, `clock_ratio`, `my_clock_frac`, `opp_clock_frac` (clock as a fraction of the base time), `rating_diff`, `material_diff` and `is_arena`.

## EDA

The full analysis is in `notebook.ipynb`. The main findings:

- Players flag more in longer time controls once they are low (36% in 3+0, 45% in 5+0, 53% in 10+0). This is because we only look at players who are already low on time. When someone drops below 10 s in a 10+0 game, their opponent usually still has minutes left (median 156 s vs 31 s in 3+0).
- Players who berserk flag less (34% vs 42% in arenas) even though they start with half the time. They are rated about 190 points higher and usually stronger than their opponent.
- The clock difference is the strongest signal. Players more than a minute behind flag 58% of the time, against 5% for players 20 to 60 s ahead.
- Move speed matters a lot. Players averaging under 1 s per move over their last 5 moves flag 16% of the time, against 59% above 15 s.
- The more pieces are left on the board, the more players flag (17% with only pawns, 50% with 9 pieces or more).
- Openings (family, ECO letter, gambits) had no effect on the validation AUC, so we dropped them.

## Models

Split 60/20/20 by `game_id`, so all the snapshots of a game end up in the same set. Each model was tuned on the validation set:

| Model | Best parameters | Val AUC |
|---|---|---|
| Logistic regression | C=0.1 | 0.783 |
| Decision tree | max_depth=15, min_samples_leaf=200 | 0.782 |
| Random forest | max_depth=15, min_samples_leaf=5, n_estimators=200 | 0.798 |
| **XGBoost** | eta=0.05, max_depth=4, min_child_weight=50, 357 rounds | **0.803** |

The final XGBoost model, retrained on train + val, gets **0.807 AUC on the test set**. It ranks best at 10 s (0.813) and worst at 60 s (0.766). The most important features are `clock_ratio`, `my_clock`, `clock_diff`, `opp_material`, `material_diff` and `my_avg_move_time`.

## How to run

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync                           # creates .venv with all dependencies
uv run jupyter lab                # opens the notebook
uv run python train.py            # trains the final model, saves model.bin
uv run python predict.py          # starts the web service on port 9696
```

Example request:

```bash
curl -X POST localhost:9696/predict -H "Content-Type: application/json" -d '{
  "threshold": 20, "event": "Rated Blitz game", "time_control": 300, "ply": 55,
  "is_white": true, "my_clock": 12.0, "opp_clock": 156.0, "my_rating": 1425,
  "opp_rating": 1458, "my_material": 21, "opp_material": 14, "my_berserk": false,
  "opp_berserk": false, "my_avg_move_time": 8.2, "opp_avg_move_time": 2.6,
  "n_pieces": 7, "queens_on": false
}'
```

```json
{"flag_probability": 0.8577, "will_flag": true}
```

### Replay real games

With the service running, `replay.py` downloads games from Lichess and sends every snapshot to the service:

```bash
uv run python replay.py https://lichess.org/<game id>
uv run python replay.py --user <lichess username> --max 10
```

Here are two of my games. In the first one, my opponent's flag probability goes up as they get lower than me (0.18 at 60 s, 0.76 at 10 s) while mine stays low, and they flag. In the second one I'm behind on the clock the whole game and the model sees it coming from 60 s (0.46).

```
havannna (2039) vs HighFlip (2089), 180+0, 0-1 (Time forfeit)  https://lichess.org/game/eJ3qI8WM
  move  player               below  clock    opp  P(flag)  flagged
    29  HighFlip               60s     58     92     0.33  no
    37  havannna               60s     58     51     0.18  yes
    40  HighFlip               40s     34     41     0.29  no
    41  havannna               40s     37     34     0.32  yes
    52  havannna               20s     15     25     0.64  yes
    54  HighFlip               20s     19     12     0.09  no
    56  havannna               10s      8     19     0.76  yes

HighFlip (2079) vs trebla1991 (2074), 180+0, 0-1 (Time forfeit)  https://lichess.org/game/Y02AKV8s
  move  player               below  clock    opp  P(flag)  flagged
    22  trebla1991             60s     58     64     0.33  no
    23  HighFlip               60s     39     58     0.46  yes
    23  HighFlip               40s     39     58     0.46  yes
    29  trebla1991             40s     36     24     0.18  no
    32  HighFlip               20s     13     30     0.60  yes
    35  HighFlip               10s      9     23     0.65  yes
    35  trebla1991             20s     19      9     0.17  no
    45  trebla1991             10s      9      2     0.07  no
```

### Docker

The image trains the model while it builds (`model.bin` is not in git), then serves it on port 9696:

```bash
docker build -t lichess-flag-prediction .
docker run -it --rm -p 9696:9696 lichess-flag-prediction
```

The curl request and `replay.py` above work the same way against the container.

## Files

| File | |
|---|---|
| `prepare_data.py` | Downloads the Lichess games and builds `data/snapshots.csv` |
| `notebook.ipynb` | Data cleaning, EDA, model tuning and selection |
| `features.py` | Feature engineering shared by `train.py` and `predict.py` |
| `train.py` | Trains the final model and saves `model.bin` |
| `predict.py` | FastAPI web service |
| `replay.py` | Replays Lichess games through the web service |
| `Dockerfile` | Image that trains the model and runs the web service |
