import io
from collections import deque
import requests
import zstandard
import chess
import chess.pgn
import argparse
import pandas as pd

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst"
TIME_CONTROL_FILTER = [(180, 0), (300, 0), (600, 0)]
TERMINATION_FILTER = ["Normal", "Time forfeit"]
THRESHOLDS = [60, 40, 20, 10] # seconds: one snapshot per player the first time their clock drops below each
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
RECENT_MOVES = 5 # number of a player's last moves used for their average move time

def material(board, color):
    return sum(len(board.pieces(piece, color)) * value for piece, value in PIECE_VALUES.items())

def avg(values):
    return sum(values) / len(values) if values else None

def n_pieces(board):
    """Knights, bishops, rooks and queens left on the board, both colors."""
    return sum(len(board.pieces(piece, color))
               for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
               for color in (chess.WHITE, chess.BLACK))

def stream_games(n_games):
    with requests.get(URL, stream=True) as response:
        response.raise_for_status()
        raw = zstandard.ZstdDecompressor().stream_reader(response.raw)
        text = io.TextIOWrapper(raw, encoding="utf-8")
        for _ in range(n_games):
            game = chess.pgn.read_game(text)
            if game is None:
                break
            yield game

def snapshots(game):
    """Return a list of rows: one per (player, threshold) the first time that player's clock drops below it."""
    tc = game.headers["TimeControl"]
    if tc == "-": # Correspondance time control
        return []

    base, increment = map(int, tc.split("+"))
    if (base, increment) not in TIME_CONTROL_FILTER:
        return []

    termination = game.headers["Termination"]
    if termination not in TERMINATION_FILTER: # Abandoned, Unterminated, ...
        return []

    result = game.headers["Result"]
    flagged_player = None
    if termination == "Time forfeit" and result != "1/2-1/2": # draw on time = insufficient material
        flagged_player = chess.BLACK if result == "1-0" else chess.WHITE

    # Replay the game, recording a snapshot each time a player crosses one of the thresholds
    rows = []
    board = game.board()
    clocks = {chess.WHITE: base, chess.BLACK: base}
    crossed = {chess.WHITE: set(), chess.BLACK: set()} # thresholds already recorded per player
    berserk = {chess.WHITE: False, chess.BLACK: False} # berserk (arena only) halves the starting clock
    move_times = {chess.WHITE: deque(maxlen=RECENT_MOVES), chess.BLACK: deque(maxlen=RECENT_MOVES)}
    for ply, node in enumerate(game.mainline(), start=1):
        player = board.turn # who is making this move
        clock = node.clock() # mover's time left after this move
        if clock is None:
            return []

        board.push(node.move)
        if ply <= 2: # each player's first move shows their starting clock
            berserk[player] = clock < base * 0.75
        else: # no increment, so the time spent is just the clock difference (+15 s gifts can make it negative)
            move_times[player].append(max(0.0, clocks[player] - clock))
        clocks[player] = clock

        for threshold in THRESHOLDS:
            if clock >= threshold or threshold in crossed[player]:
                continue
            crossed[player].add(threshold)

            opponent = not player
            is_white = player == chess.WHITE
            rows.append({
                "game_id": game.headers["Site"].rsplit("/", 1)[-1],
                "threshold": threshold,
                "event": game.headers["Event"],
                "eco": game.headers["ECO"],
                "opening": game.headers["Opening"],
                "time_control": base,
                "ply": ply,
                "is_white": is_white,
                "my_clock": clocks[player],
                "opp_clock": clocks[opponent],
                "my_rating": int(game.headers["WhiteElo" if is_white else "BlackElo"]),
                "opp_rating": int(game.headers["BlackElo" if is_white else "WhiteElo"]),
                "my_material": material(board, player),
                "opp_material": material(board, opponent),
                "my_berserk": berserk[player],
                "opp_berserk": berserk[opponent],
                "my_avg_move_time": avg(move_times[player]),
                "opp_avg_move_time": avg(move_times[opponent]),
                "n_pieces": n_pieces(board),
                "queens_on": bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK)),
                "flagged": int(player == flagged_player),
            })

    return rows

def save_snapshots(n_games, output):
    rows = []
    for game in stream_games(n_games):
        rows.extend(snapshots(game))

    pd.DataFrame(rows).to_csv(output, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--games", type=int, default=200_000)
    parser.add_argument("-o", "--output", default="data/snapshots.csv")
    args = parser.parse_args()

    save_snapshots(args.games, args.output)

if __name__ == "__main__":
    main()