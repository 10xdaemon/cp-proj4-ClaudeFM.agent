"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


PROFILES = {
    "High-Energy Pop": {
        "preferred_mood":     "happy",
        "preferred_genre":    "pop",
        "target_energy":       0.88,
        "target_tempo_bpm":    125,
        "target_acousticness": 0.10,
        "target_speechiness":  0.06,
        "sigma":               0.20,
    },
    "Chill Lofi": {
        "preferred_mood":     "chill",
        "preferred_genre":    "lofi",
        "target_energy":       0.38,
        "target_tempo_bpm":    76,
        "target_acousticness": 0.80,
        "target_speechiness":  0.02,
        "sigma":               0.20,
    },
    "Deep Intense Rock": {
        "preferred_mood":     "intense",
        "preferred_genre":    "rock",
        "target_energy":       0.93,
        "target_tempo_bpm":    152,
        "target_acousticness": 0.10,
        "target_speechiness":  0.06,
        "sigma":               0.15,
    },
}


def print_recommendations(label: str, recommendations: list) -> None:
    print("\n" + "=" * 50)
    print(f"  {label}  —  Top {len(recommendations)} Recommendations")
    print("=" * 50)
    for i, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n#{i}  {song['title']}  —  {song['artist']}")
        print(f"    Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"    Score: {score:.2f} / 8.0")
        print("    Reasons:")
        for reason in explanation.split(' | '):
            print(f"      • {reason}")
    print("\n" + "=" * 50)


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for label, user_prefs in PROFILES.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(label, recommendations)


if __name__ == "__main__":
    main()
