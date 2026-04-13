"""
Music recommendation engine.

Provides load_songs, score_song, and recommend_songs for scoring and
ranking a catalog of songs against a user preference profile.
"""

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """Represents a song and its audio feature attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    speechiness: float
    instrumentalness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences used to score songs."""
    preferred_mood: str
    preferred_genre: str
    target_energy: float
    target_tempo_bpm: float
    target_acousticness: float
    target_speechiness: float
    sigma: float


class Recommender:
    """OOP wrapper around the recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Adjacency graph used for mood distance scoring.
# Edges are undirected: if B is in MOOD_GRAPH[A], then A is in MOOD_GRAPH[B].
MOOD_GRAPH: Dict[str, set] = {
    'euphoric':    {'happy'},
    'happy':       {'euphoric', 'uplifted', 'relaxed'},
    'uplifted':    {'happy', 'groovy'},
    'groovy':      {'uplifted', 'bittersweet'},
    'relaxed':     {'happy', 'romantic', 'chill'},
    'romantic':    {'relaxed', 'bittersweet'},
    'bittersweet': {'groovy', 'romantic'},
    'chill':       {'relaxed', 'nostalgic', 'focused'},
    'nostalgic':   {'chill', 'peaceful'},
    'peaceful':    {'nostalgic'},
    'focused':     {'chill', 'moody', 'intense'},
    'moody':       {'focused', 'melancholic', 'dark'},
    'melancholic': {'moody'},
    'intense':     {'focused', 'dark'},
    'dark':        {'moody', 'intense', 'angry'},
    'angry':       {'dark'},
}

# BPM range used to normalise tempo before Gaussian scoring.
_TEMPO_MIN = 52.0
_TEMPO_MAX = 168.0

# Maximum points awarded per scoring dimension (sum = 8.0).
_W_GENRE        = 2.0
_W_ENERGY       = 2.0
_W_ACOUSTICNESS = 1.5
_W_TEMPO        = 1.0
_W_SPEECHINESS  = 0.5

# Mood graph distance → points awarded.
_MOOD_POINTS: Dict[int, float] = {0: 1.0, 1: 0.5, 2: 0.2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(diff: float, sigma: float) -> float:
    """Return a Gaussian similarity score in (0, 1] for a given difference."""
    return math.exp(-(diff ** 2) / (2 * sigma ** 2))


def _mood_distance(user_mood: str, song_mood: str) -> int:
    """Return the graph distance between two moods (0, 1, 2, or 99 if unreachable)."""
    if user_mood == song_mood:
        return 0
    neighbors = MOOD_GRAPH.get(user_mood, set())
    if song_mood in neighbors:
        return 1
    if any(song_mood in MOOD_GRAPH.get(n, set()) for n in neighbors):
        return 2
    return 99


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Read a songs CSV file and return a list of typed song dicts."""
    songs = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            songs.append({
                'id':               int(clean['id']),
                'title':            clean['title'],
                'artist':           clean['artist'],
                'genre':            clean['genre'],
                'mood':             clean['mood'],
                'energy':           float(clean['energy']),
                'tempo_bpm':        float(clean['tempo_bpm']),
                'valence':          float(clean['valence']),
                'danceability':     float(clean['danceability']),
                'acousticness':     float(clean['acousticness']),
                'speechiness':      float(clean['speechiness']),
                'instrumentalness': float(clean['instrumentalness']),
            })
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences.

    Returns:
        A (total_score, reasons) tuple where reasons is a list of
        human-readable strings describing each scoring component.
    """
    sigma = user_prefs.get('sigma', 0.20)
    score = 0.0
    reasons = []

    # Genre match: max +2.0
    if song['genre'] == user_prefs.get('preferred_genre', ''):
        score += _W_GENRE
        reasons.append(f'genre match (+{_W_GENRE})')

    # Mood adjacency: +1.0 exact / +0.5 one step / +0.2 two steps
    dist = _mood_distance(user_prefs.get('preferred_mood', ''), song['mood'])
    mood_pts = _MOOD_POINTS.get(dist, 0.0)
    if mood_pts > 0:
        label = 'match' if dist == 0 else f'{dist}-step away'
        score += mood_pts
        reasons.append(f'mood {label} (+{mood_pts})')

    # Continuous feature proximity via Gaussian (weighted by importance)
    norm_user_tempo = (user_prefs['target_tempo_bpm'] - _TEMPO_MIN) / (_TEMPO_MAX - _TEMPO_MIN)
    norm_song_tempo = (song['tempo_bpm']              - _TEMPO_MIN) / (_TEMPO_MAX - _TEMPO_MIN)

    components = (
        (_W_ENERGY,       abs(user_prefs['target_energy']       - song['energy']),       'energy proximity'),
        (_W_ACOUSTICNESS, abs(user_prefs['target_acousticness'] - song['acousticness']), 'acousticness proximity'),
        (_W_TEMPO,        abs(norm_user_tempo - norm_song_tempo),                        'tempo proximity'),
        (_W_SPEECHINESS,  abs(user_prefs['target_speechiness']  - song['speechiness']),  'speechiness proximity'),
    )
    for weight, diff, label in components:
        pts = weight * _gaussian(diff, sigma)
        score += pts
        reasons.append(f'{label} (+{pts:.2f})')

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song, sort by score descending, and return the top k results.

    A genre-streak cap (max 2 consecutive same-genre songs) is applied to
    encourage variety in the final list.
    """
    scored = []
    for song in songs:
        song_score, reasons = score_song(user_prefs, song)
        scored.append((song, song_score, ' | '.join(reasons)))
    scored.sort(key=lambda x: x[1], reverse=True)

    results: List[Tuple[Dict, float, str]] = []
    last_genre: str | None = None
    streak = 0

    for song, song_score, explanation in scored:
        if song['genre'] == last_genre:
            streak += 1
        else:
            last_genre = song['genre']
            streak = 1
        if streak <= 2:
            results.append((song, song_score, explanation))
        if len(results) == k:
            break

    return results
