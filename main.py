from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from textblob import TextBlob
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAI
client_ai = OpenAI(api_key="YOUR_API_KEY")

load_dotenv()


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL")
client = MongoClient(MONGO_URL)

client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
db = client["sentimentdb"]
collection = db["entries"]

# Sentiment function
def analyze(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    text_lower = text.lower()

    # 🔥 Boost weak positive words
    positive_boost = ["like", "nice", "good", "okay", "fine", "cool", "happy", "enjoy"]
    
    # 🔥 Boost strong positive words
    strong_positive = ["love", "amazing", "awesome", "great", "excellent", "fantastic"]

    # 🔥 Boost negative words
    negative_boost = ["bad", "sad", "dislike", "poor", "annoying"]

    # 🔥 Boost strong negative words
    strong_negative = ["hate", "terrible", "awful", "worst", "horrible"]

    # Apply boosts
    for word in positive_boost:
        if word in text_lower:
            polarity += 0.2

    for word in strong_positive:
        if word in text_lower:
            polarity += 0.4

    for word in negative_boost:
        if word in text_lower:
            polarity -= 0.2

    for word in strong_negative:
        if word in text_lower:
            polarity -= 0.4

    # Final classification
    if polarity > 0.05:
        return "Positive", polarity
    elif polarity < -0.05:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    data = list(collection.find().sort("timestamp", -1))
    
    pos = collection.count_documents({"sentiment": "Positive"})
    neg = collection.count_documents({"sentiment": "Negative"})
    neu = collection.count_documents({"sentiment": "Neutral"})
    twitter_count = collection.count_documents({"source": "Twitter"})
    reddit_count = collection.count_documents({"source": "Reddit"})
    insta_count = collection.count_documents({"source": "Instagram"})
    # Get most positive & negative entries
    top_positive = collection.find_one(sort=[("score", -1)])
    top_negative = collection.find_one(sort=[("score", 1)])

    return templates.TemplateResponse("index.html", {
    "request": request,
    "entries": data,
    "pos": pos,
    "neg": neg,
    "neu": neu,
    "twitter": twitter_count,
    "reddit": reddit_count,
    "insta": insta_count,
    "top_positive": top_positive,
    "top_negative": top_negative
})


# Analyze text
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_text(request: Request, text: str = Form(...), source: str = Form(...)):
    
    # Step 1: Basic Sentiment Analysis
    sentiment, score = analyze(text)

    # Step 2: Determine intensity (for smarter AI context)
    if abs(score) > 0.5:
        intensity = "strong"
    elif abs(score) > 0.2:
        intensity = "moderate"
    else:
        intensity = "weak"

    # Step 3: Advanced AI Insight (Improved Prompt)
    response = client_ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert sentiment analyst. Provide meaningful, concise, and structured insights."
            },
            {
                "role": "user",
                "content": f"""
Analyze the following text and provide output in EXACT format:

Insight: (1 short sentence summary)
Reason: (why this sentiment was detected)
Suggestion: (what this implies or what action can be taken)

Text: "{text}"
Detected Sentiment: {sentiment}
Score: {round(score, 3)}
Intensity: {intensity}
"""
            }
        ],
        temperature=0.7
    )

    ai_explanation = response.choices[0].message.content

    # Step 4: Store in MongoDB
    collection.insert_one({
        "text": text,
        "source": source,
        "sentiment": sentiment,
        "score": round(score, 3),
        "ai": ai_explanation,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    # Step 5: Return updated page
    return await home(request)