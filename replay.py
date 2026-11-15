"""Replay Lichess games through the web service.

https://lichess.org/api#tag/games/GET/game/export/{gameId}
"""

import argparse
import io
import chess.pgn
import requests

from prepare_data import snapshots

LICHESS_URL = "https://lichess.org"
PGN_PARAMS = {"clocks": "true", "opening": "true"} # clock comments and ECO/Opening headers are needed
HEADERS = {"User-Agent": "lichess-flag-prediction"} # Lichess refuses some requests without one


def fetch_game(game):
    parts = game.rstrip("/").split("/") # accepts an id or a url, a url can end with /white or /black
    game_id = (parts[-2] if parts[-1] in ("white", "black") else parts[-1])[:8]
    response = requests.get(f"{LICHESS_URL}/game/export/{game_id}", params=PGN_PARAMS, headers=HEADERS)
    response.raise_for_status()
    return response.text


def fetch_user_games(user, max_games):
    params = {**PGN_PARAMS, "max": max_games, "perfType": "blitz,rapid"}
    response = requests.get(f"{LICHESS_URL}/api/games/user/{user}", params=params, headers=HEADERS)
    response.raise_for_status()
    return response.text


def read_games(pgn_text):
    pgn = io.StringIO(pgn_text)
    while (game := chess.pgn.read_game(pgn)) is not None:
        yield game


def replay(game, url):
    headers = game.headers
    print(f"\n{headers['White']} ({headers['WhiteElo']}) vs {headers['Black']} ({headers['BlackElo']}), "
          f"{headers['TimeControl']}, {headers['Result']} ({headers['Termination']})  {headers['Site']}")

    rows = snapshots(game)
    if not rows:
        return

    print(f"  {'move':>4}  {'player':<20} {'below':>5}  {'clock':>5}  {'opp':>5}  {'P(flag)':>7}  flagged")
    for row in rows:
        flagged = row.pop("flagged")
        row.pop("game_id")
        prediction = requests.post(url, json=row).json()
        player = headers["White"] if row["is_white"] else headers["Black"]
        print(f"  {(row['ply'] + 1) // 2:>4}  {player:<20} {row['threshold']:>4}s  {row['my_clock']:>5.0f}  "
              f"{row['opp_clock']:>5.0f}  {prediction['flag_probability']:>7.2f}  {'yes' if flagged else 'no'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("games", nargs="*", help="game ids or urls")
    parser.add_argument("-u", "--user", help="replay this user's last blitz/rapid games")
    parser.add_argument("-m", "--max", type=int, default=10, help="number of games with --user")
    parser.add_argument("--url", default="http://localhost:9696/predict")
    args = parser.parse_args()

    pgn_texts = [fetch_game(game) for game in args.games]
    if args.user:
        pgn_texts.append(fetch_user_games(args.user, args.max))

    for pgn_text in pgn_texts:
        for game in read_games(pgn_text):
            replay(game, args.url)


if __name__ == "__main__":
    main()
