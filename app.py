import streamlit as st
from src.agent import run_agent, AgentResult
from src.spotify_client import get_token, fetch_recommendations
from src.scorer import gaussian_score_normalized, blend
from src.recommender import UserProfile

st.set_page_config(page_title="AI Playlist Builder", page_icon="🎵", layout="wide")

# ── Session state initialization ───────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_weight" not in st.session_state:
    st.session_state.last_weight = 0.5
if "extra_songs" not in st.session_state:
    st.session_state.extra_songs = []
if "extra_scores" not in st.session_state:
    st.session_state.extra_scores = []
if "seeds" not in st.session_state:
    st.session_state.seeds = {"artists": [], "genres": []}
if "profile" not in st.session_state:
    st.session_state.profile = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    gaussian_weight = st.slider(
        "Algorithm  ←→  AI",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 = full LLM scoring · 1 = full Gaussian scoring",
    )

    st.markdown("---")
    st.subheader("Preset Queries")

    if st.button("Study session"):
        st.session_state.query = "create a playlist for me to study to"
        st.rerun()
    if st.button("Party vibes"):
        st.session_state.query = "I'm at a party, play something energetic"
        st.rerun()
    if st.button("Late-night drive"):
        st.session_state.query = "make me a playlist for a late-night drive"
        st.rerun()

    st.markdown("---")
    st.caption("API keys loaded from .env")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🎵 AI Playlist Builder")
st.caption("Type what you're about to do and get a personalized playlist.")

query = st.text_area(
    "What are you up to?",
    key="query",
    placeholder="e.g. 'create a playlist for studying' or 'my friend loves Bad Bunny, we're at a party'",
    height=80,
)

submitted = st.button("Build Playlist", type="primary")

if submitted and query.strip():
    cache_miss = (
        query != st.session_state.last_query
        or gaussian_weight != st.session_state.last_weight
    )
    if cache_miss:
        with st.spinner("Building your playlist..."):
            st.session_state.result = run_agent(query, gaussian_weight)
            st.session_state.last_query = query
            st.session_state.last_weight = gaussian_weight
            st.session_state.extra_songs = []
            st.session_state.extra_scores = []
            # Extract seeds and profile from reasoning steps for Load More
            for step in st.session_state.result.reasoning_steps:
                if step["tool"] == "parse_user_intent":
                    inp = step["input"]
                    st.session_state.seeds = {
                        "artists": inp.get("seed_artists", []),
                        "genres": inp.get("seed_genres", []),
                    }
                    st.session_state.profile = UserProfile(
                        preferred_mood=inp["preferred_mood"],
                        preferred_genre=inp["preferred_genre"],
                        target_energy=float(inp["target_energy"]),
                        target_tempo_bpm=float(inp["target_tempo_bpm"]),
                        target_acousticness=float(inp["target_acousticness"]),
                        target_speechiness=float(inp["target_speechiness"]),
                        sigma=float(inp["sigma"]),
                    )

elif submitted and not query.strip():
    st.warning("Please enter a query before building a playlist.")

# ── Results ───────────────────────────────────────────────────────────────────
result: AgentResult | None = st.session_state.result
if result is not None:
    # ── Guardrail warnings ──────────────────────────────────────────────────
    if result.guardrail_warnings:
        for w in result.guardrail_warnings:
            st.warning(w)

    # ── Agent reasoning trace ────────────────────────────────────────────────
    with st.expander("Agent Reasoning", expanded=False):
        for step in result.reasoning_steps:
            st.markdown(f"**`{step['tool']}`**")
            st.json(step["input"], expanded=False)
            st.caption(str(step["output"])[:300])
            st.divider()

    # ── Confidence badge ─────────────────────────────────────────────────────
    conf_label = (
        "🟢 High" if result.confidence >= 0.7
        else ("🟡 Medium" if result.confidence >= 0.3 else "🔴 Low")
    )
    st.metric("Confidence", conf_label, help="Gap between rank-1 and rank-2 blended score")

    st.markdown("---")

    # ── Top 3 songs: cover art + explanation ─────────────────────────────────
    st.subheader("Your Playlist")
    all_songs = result.songs + st.session_state.extra_songs
    all_scores = result.scores + st.session_state.extra_scores

    if result.songs:
        cols = st.columns(3)
        for i, (col, song, score, explanation) in enumerate(
            zip(cols, result.songs[:3], result.scores[:3], result.explanations)
        ):
            with col:
                img_url = song.cover_art_url or f"https://placehold.co/200x200?text={i + 1}"
                st.image(img_url, width=200)
                st.markdown(f"**{song.title}**")
                st.caption(f"{song.artist} · {song.genre}")
                st.progress(min(max(score, 0.0), 1.0), text=f"{score:.2f}")
                st.markdown(f"_{explanation}_")

    # ── Songs 4–10: compact list ─────────────────────────────────────────────
    if len(all_songs) > 3:
        st.markdown("#### More tracks")
        for i, (song, score) in enumerate(zip(all_songs[3:], all_scores[3:]), start=4):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{i}.** {song.title} — *{song.artist}*")
            c2.markdown(f"`{score:.2f}`")

    # ── Load More ────────────────────────────────────────────────────────────
    if st.button("Load More"):
        with st.spinner("Fetching more tracks..."):
            token = get_token()
            seeds = st.session_state.seeds
            more = fetch_recommendations(
                token=token,
                seed_artists=seeds["artists"],
                seed_genres=seeds["genres"],
                limit=10,
            )
            profile = st.session_state.profile
            if profile and more:
                g_scores = [gaussian_score_normalized(profile, s) for s in more]
                ranked = sorted(zip(more, g_scores), key=lambda x: x[1], reverse=True)
                st.session_state.extra_songs += [s for s, _ in ranked]
                st.session_state.extra_scores += [sc for _, sc in ranked]
        st.rerun()
