import csv
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """A song and its audio feature attributes."""
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
    """A user's taste preferences used to score songs."""
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

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv and return a list of typed song dicts."""
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

_TEMPO_MIN = 52.0
_TEMPO_MAX = 168.0


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences; return (total_score, reasons)."""
    sigma = user_prefs.get('sigma', 0.20)
    score = 0.0
    reasons = []

    # Genre match: +2.0
    if song['genre'] == user_prefs.get('preferred_genre', ''):
        score += 2.0
        reasons.append('genre match (+2.0)')

    # Mood adjacency: +1.0 exact / +0.5 one step / +0.2 two steps
    user_mood = user_prefs.get('preferred_mood', '')
    song_mood = song['mood']
    if user_mood == song_mood:
        dist = 0
    elif song_mood in MOOD_GRAPH.get(user_mood, set()):
        dist = 1
    elif any(song_mood in MOOD_GRAPH.get(n, set()) for n in MOOD_GRAPH.get(user_mood, set())):
        dist = 2
    else:
        dist = 99
    mood_pts = {0: 1.0, 1: 0.5, 2: 0.2}.get(dist, 0.0)
    if mood_pts > 0:
        label = 'match' if dist == 0 else f'{dist}-step away'
        score += mood_pts
        reasons.append(f'mood {label} (+{mood_pts})')

    # Energy proximity: max +2.0
    e_pts = 2.0 * math.exp(-(abs(user_prefs['target_energy'] - song['energy']) ** 2) / (2 * sigma ** 2))
    score += e_pts
    reasons.append(f'energy proximity (+{e_pts:.2f})')

    # Acousticness proximity: max +1.5
    a_pts = 1.5 * math.exp(-(abs(user_prefs['target_acousticness'] - song['acousticness']) ** 2) / (2 * sigma ** 2))
    score += a_pts
    reasons.append(f'acousticness proximity (+{a_pts:.2f})')

    # Tempo proximity (normalized 0-1): max +1.0
    norm_user = (user_prefs['target_tempo_bpm'] - _TEMPO_MIN) / (_TEMPO_MAX - _TEMPO_MIN)
    norm_song = (song['tempo_bpm'] - _TEMPO_MIN) / (_TEMPO_MAX - _TEMPO_MIN)
    t_pts = 1.0 * math.exp(-(abs(norm_user - norm_song) ** 2) / (2 * sigma ** 2))
    score += t_pts
    reasons.append(f'tempo proximity (+{t_pts:.2f})')

    # Speechiness proximity: max +0.5
    s_pts = 0.5 * math.exp(-(abs(user_prefs['target_speechiness'] - song['speechiness']) ** 2) / (2 * sigma ** 2))
    score += s_pts
    reasons.append(f'speechiness proximity (+{s_pts:.2f})')

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song, sort by score descending, return top k results."""
    # Step 1: score every song independently
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, ' | '.join(reasons)))

    # Step 2: sort the full catalog by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Step 3: walk the sorted list, capping same-genre runs for variety
    results = []
    last_genre = None
    streak = 0
    for song, score, explanation in scored:
        if song['genre'] == last_genre:
            streak += 1
        else:
            last_genre = song['genre']
            streak = 1
        if streak <= 2:
            results.append((song, score, explanation))
        if len(results) == k:
            break

    return results

# user1 = {
#     "preferred_mood": "intense",
#     "preferred_genre": "rock",
#     "target_energy": 0.93,
#     "target_tempo_bpm": 152,
#     "target_acousticness": 0.10,
#     "target_speechiness": 0.06,
#     "sigma": 0.15
# }
