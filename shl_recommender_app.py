import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 1. DATA PIPELINE (Scrape → Parse → Store)
# ---------------------------

def load_shl_catalog():
    """
    In production:
    - Scrape SHL website
    - Parse HTML
    - Store in DB / CSV
    """

    data = {
        "assessment_name": [
            "Verify Numerical Reasoning",
            "Java Programming Simulation",
            "OPQ32 Personality Questionnaire",
            "Mechanical Comprehension",
            "Customer Contact Simulation"
        ],
        "description": [
            "Measures numerical analysis and business data interpretation.",
            "Evaluates Java coding, debugging, and algorithmic thinking.",
            "Assesses workplace personality traits and behavior.",
            "Tests understanding of mechanical and physical principles.",
            "Evaluates communication, empathy, and customer handling."
        ],
        "job_level": [
            "Graduate/Professional",
            "IT / Developer",
            "All Levels",
            "Technical / Engineering",
            "Entry Level"
        ]
    }

    return pd.DataFrame(data)

df = load_shl_catalog()

# ---------------------------
# 2. RAG-STYLE RECOMMENDER
# ---------------------------

class SHLRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(
            df["description"].tolist(), normalize_embeddings=True
        )

    def recommend(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        )

        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        self.df["similarity_score"] = similarities

        results = (
            self.df.sort_values("similarity_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

        return results[
            ["assessment_name", "job_level", "similarity_score"]
        ]

recommender = SHLRecommender(df)

# ---------------------------
# 3. FASTAPI WEB APP
# ---------------------------

app = FastAPI(
    title="SHL Assessment Recommendation Engine",
    description="Semantic RAG-based recommender for SHL product catalog",
    version="1.0"
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/recommend")
def recommend_assessment(request: QueryRequest):
    results = recommender.recommend(request.query, request.top_k)
    return {
        "query": request.query,
        "recommendations": results.to_dict(orient="records")
    }

# ---------------------------
# 4. EVALUATION METHOD
# ---------------------------

def evaluate_precision_at_k(sample_queries):
    correct = 0
    total = len(sample_queries)

    for query, expected in sample_queries:
        result = recommender.recommend(query, top_k=1)
        if expected in result.iloc[0]["assessment_name"]:
            correct += 1

    return correct / total
