"""Optional AI interpretation layer for CineMind.

If OPENAI_API_KEY is configured, an OpenAI model converts natural language into a
small JSON preference object. Without a key, a deterministic local parser keeps the
feature usable for demos and deployment without paid API access.
"""
import json, os, re
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GENRES = ["Action","Adventure","Animation","Children","Comedy","Crime","Drama","Fantasy","Horror","Romance","Sci-Fi","Thriller","Musical","Mystery","War","Western"]
KEYWORD_MAP = {
    "funny":"Comedy", "comedy":"Comedy", "romantic":"Romance", "romance":"Romance",
    "love":"Romance", "scary":"Horror", "horror":"Horror", "thriller":"Thriller",
    "suspense":"Thriller", "crime":"Crime", "gangster":"Crime", "space":"Sci-Fi",
    "sci-fi":"Sci-Fi", "science fiction":"Sci-Fi", "superhero":"Action", "action":"Action",
    "adventure":"Adventure", "fantasy":"Fantasy", "animated":"Animation", "animation":"Animation",
    "family":"Children", "war":"War", "western":"Western", "musical":"Musical", "mystery":"Mystery",
}

def _local_parse(text, known_titles):
    low=text.lower(); genres=[]
    for key,g in KEYWORD_MAP.items():
        if key in low and g not in genres: genres.append(g)
    seeds=[]
    # Match complete known titles from the catalogue, longest first.
    for t in sorted(known_titles, key=len, reverse=True):
        if t.lower() in low:
            seeds.append(t)
            if len(seeds)>=3: break
    mood=[]
    for k in ["dark","lighthearted","emotional","feel-good","mind-bending","fast-paced","serious","chill","intense","funny","romantic"]:
        if k in low: mood.append(k)
    avoid=[]
    if any(x in low for x in ["no horror","without horror","not horror"]): avoid.append("Horror")
    if any(x in low for x in ["no romance","without romance","not romantic"]): avoid.append("Romance")
    return {"genres":genres,"keywords":re.findall(r"\b[a-z]{4,}\b", low)[:12],"mood":mood,"seed_titles":seeds,"avoid_genres":avoid}

def parse_preferences(text, known_titles):
    key=os.getenv("OPENAI_API_KEY")
    if not key:
        return _local_parse(text, known_titles), "local"
    try:
        from openai import OpenAI
        client=OpenAI(api_key=key)
        schema={"type":"object","properties":{
            "genres":{"type":"array","items":{"type":"string"}},
            "keywords":{"type":"array","items":{"type":"string"}},
            "mood":{"type":"array","items":{"type":"string"}},
            "seed_titles":{"type":"array","items":{"type":"string"}},
            "avoid_genres":{"type":"array","items":{"type":"string"}}
        },"required":["genres","keywords","mood","seed_titles","avoid_genres"],"additionalProperties":False}
        response=client.responses.create(
            model=os.getenv("OPENAI_MODEL","gpt-5-mini"),
            input=[{"role":"system","content":"You are CineMind's movie-preference parser. Return only JSON. Map the user's request to movie genres, descriptive keywords, mood, up to 3 exact seed titles from the supplied catalogue, and genres to avoid."},
                   {"role":"user","content":f"Catalogue titles: {', '.join(known_titles[:1500])}\n\nRequest: {text}"}],
            text={"format":{"type":"json_schema","name":"movie_preferences","schema":schema,"strict":True}},
        )
        data=json.loads(response.output_text)
        # Keep only catalogue titles and known genres.
        data["genres"]=[g for g in data.get("genres",[]) if g in GENRES]
        data["seed_titles"]=[t for t in data.get("seed_titles",[]) if t in known_titles][:3]
        return data, "openai"
    except Exception:
        return _local_parse(text, known_titles), "local-fallback"
