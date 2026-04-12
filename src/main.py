"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Starter example profile
    user_prefs = {
        "preferred_mood":     "happy",
        "preferred_genre":    "pop",
        "target_energy":       0.80,
        "target_tempo_bpm":    118,
        "target_acousticness": 0.18,
        "target_speechiness":  0.05,
        "sigma":               0.20,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 50)
    print(f"  Top {len(recommendations)} Recommendations")
    print("=" * 50)

    for i, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n#{i}  {song['title']}  —  {song['artist']}")
        print(f"    Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"    Score: {score:.2f} / 8.0")
        print("    Reasons:")
        for reason in explanation.split(' | '):
            print(f"      • {reason}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
