import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import User as UserSchema, Profile as ProfileSchema, Food as FoodSchema, Consumption as ConsumptionSchema, Weight as WeightSchema, GroceryList as GroceryListSchema, Planner as PlannerSchema, Badge as BadgeSchema

app = FastAPI(title="HLife API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Utilities --------
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


def oid_str(oid: Any) -> str:
    try:
        return str(oid)
    except Exception:
        return oid


def doc_to_public(d: dict) -> dict:
    d = {**d}
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # convert ObjectIds nested
    for k, v in list(d.items()):
        if isinstance(v, ObjectId):
            d[k] = str(v)
    return d


# -------- Auth & Profile --------
class AuthRegister(BaseModel):
    name: str
    email: EmailStr
    password: str


class AuthLogin(BaseModel):
    email: EmailStr
    password: str


def hash_password(pw: str) -> str:
    import hashlib
    return hashlib.sha256(pw.encode()).hexdigest()


@app.post("/auth/register")
def register(payload: AuthRegister):
    existing = db["user"].find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = UserSchema(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        is_premium=False,
    )
    user_id = create_document("user", user)
    # create empty profile
    prof = ProfileSchema(user_id=user_id)
    create_document("profile", prof)
    return {"id": user_id, "name": user.name, "email": user.email, "is_premium": False}


@app.post("/auth/login")
def login(payload: AuthLogin):
    user = db["user"].find_one({"email": payload.email})
    if not user or user.get("password_hash") != hash_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    u = doc_to_public(user)
    # Note: For simplicity we return user data without JWT session management
    return {"id": u["id"], "name": u["name"], "email": u["email"], "is_premium": bool(u.get("is_premium", False))}


class ProfileUpdate(BaseModel):
    allergies: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    preferences: Optional[Dict[str, List[str]]] = None
    calorie_target: Optional[int] = None
    protein_target: Optional[int] = None


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    prof = db["profile"].find_one({"user_id": user_id})
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found")
    return doc_to_public(prof)


@app.put("/profile/{user_id}")
def update_profile(user_id: str, payload: ProfileUpdate):
    updates = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
    db["profile"].update_one({"user_id": user_id}, {"$set": updates}, upsert=True)
    prof = db["profile"].find_one({"user_id": user_id})
    return doc_to_public(prof)


# -------- Foods --------
class FoodCreate(BaseModel):
    name: str
    description: Optional[str] = None
    nutrients: Dict[str, float] = Field(default_factory=dict)
    ingredients: List[str] = Field(default_factory=list)
    recipe: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    suitable_for: List[str] = Field(default_factory=list)


@app.post("/foods")
def create_food(payload: FoodCreate):
    food = FoodSchema(**payload.model_dump())
    fid = create_document("food", food)
    return {"id": fid}


@app.get("/foods")
def list_foods(q: Optional[str] = None, tag: Optional[str] = None, limit: int = 50):
    filt: Dict[str, Any] = {}
    if q:
        filt["name"] = {"$regex": q, "$options": "i"}
    if tag:
        filt["tags"] = tag
    docs = db["food"].find(filt).limit(limit)
    return [doc_to_public(d) for d in docs]


@app.get("/foods/{food_id}")
def get_food(food_id: str):
    try:
        d = db["food"].find_one({"_id": ObjectId(food_id)})
    except Exception:
        d = None
    if not d:
        raise HTTPException(status_code=404, detail="Food not found")
    item = doc_to_public(d)
    # add shopping links for ingredients
    base = "https://www.google.com/search?q="
    links = {ing: f"{base}{ing.replace(' ', '+')}" for ing in item.get("ingredients", [])}
    item["ingredient_links"] = links
    return item


# -------- Consumption --------
class ConsumptionCreate(BaseModel):
    user_id: str
    food_id: Optional[str] = None
    custom_name: Optional[str] = None
    portion_size: float = 1.0


@app.post("/consumptions")
def create_consumption(payload: ConsumptionCreate):
    nutrients: Dict[str, float] = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    if payload.food_id:
        f = db["food"].find_one({"_id": ObjectId(payload.food_id)})
        if not f:
            raise HTTPException(status_code=404, detail="Food not found")
        for k in nutrients.keys():
            base = float(f.get("nutrients", {}).get(k, 0.0))
            nutrients[k] = round(base * float(payload.portion_size), 2)
    cons = ConsumptionSchema(
        user_id=payload.user_id,
        food_id=payload.food_id,
        custom_name=payload.custom_name,
        portion_size=payload.portion_size,
        nutrients=nutrients,
    )
    cid = create_document("consumption", cons)
    return {"id": cid}


@app.get("/consumptions/{user_id}")
def list_consumptions(user_id: str, days: int = 7):
    since = datetime.utcnow() - timedelta(days=days)
    docs = db["consumption"].find({"user_id": user_id, "consumed_at": {"$gte": since}}).sort("consumed_at", -1)
    return [doc_to_public(d) for d in docs]


# -------- Weights & Badges --------
class WeightCreate(BaseModel):
    user_id: str
    value: float


@app.post("/weights")
def add_weight(payload: WeightCreate):
    wid = create_document("weight", WeightSchema(user_id=payload.user_id, value=payload.value))
    return {"id": wid}


@app.get("/weights/{user_id}")
def list_weights(user_id: str, days: int = 90):
    since = datetime.utcnow() - timedelta(days=days)
    docs = db["weight"].find({"user_id": user_id, "recorded_at": {"$gte": since}}).sort("recorded_at", 1)
    return [doc_to_public(d) for d in docs]


@app.get("/badges/{user_id}")
def get_badges(user_id: str):
    # Simple rule: 7-day streak
    since = datetime.utcnow() - timedelta(days=7)
    count = db["consumption"].count_documents({"user_id": user_id, "consumed_at": {"$gte": since}})
    badges: List[dict] = []
    if count >= 7:
        badges.append({"key": "streak7", "label": "7-day Tracker", "awarded_at": datetime.utcnow()})
    return [doc_to_public(b) for b in badges]


# -------- Analytics --------
@app.get("/analytics/nutrition/{user_id}")
def nutrition_analytics(user_id: str, period: str = Query("weekly", pattern="^(weekly|monthly)$")):
    days = 7 if period == "weekly" else 30
    since = datetime.utcnow() - timedelta(days=days)
    docs = db["consumption"].find({"user_id": user_id, "consumed_at": {"$gte": since}})
    # Aggregate by day
    by_day: Dict[str, Dict[str, float]] = {}
    for d in docs:
        day = d["consumed_at"].strftime("%Y-%m-%d")
        by_day.setdefault(day, {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0})
        for k in ["calories", "protein", "carbs", "fat"]:
            by_day[day][k] += float(d.get("nutrients", {}).get(k, 0.0))
    # transform to list sorted by date
    items = [
        {"date": k, **{m: round(v[m], 2) for m in ["calories", "protein", "carbs", "fat"]}}
        for k, v in sorted(by_day.items())
    ]
    return {"period": period, "days": items}


# -------- Calculators --------
class BMICalcRequest(BaseModel):
    height_cm: float
    weight_kg: float


@app.post("/calc/bmi")
def calc_bmi(req: BMICalcRequest):
    h = req.height_cm / 100.0
    bmi = req.weight_kg / (h * h)
    return {"bmi": round(bmi, 2)}


class CalorieCalcRequest(BaseModel):
    height_cm: float
    weight_kg: float
    age: int
    sex: str = Field(..., pattern="^(male|female)$")
    activity: str = Field("moderate", pattern="^(sedentary|light|moderate|active|very_active)$")


@app.post("/calc/calories")
def calc_calories(req: CalorieCalcRequest):
    # Mifflin-St Jeor
    s = 5 if req.sex == "male" else -161
    bmr = 10 * req.weight_kg + 6.25 * req.height_cm - 5 * req.age + s
    factor = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }[req.activity]
    tdee = bmr * factor
    return {"bmr": round(bmr, 1), "tdee": round(tdee, 1)}


# -------- Recommendations --------
@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, limit: int = 3):
    profile = db["profile"].find_one({"user_id": user_id}) or {}
    liked = set((profile.get("preferences", {}) or {}).get("liked", []))
    disliked = set((profile.get("preferences", {}) or {}).get("disliked", []))
    conditions = set(profile.get("conditions", []) or [])

    # score foods by matching tags and suitability
    foods = list(db["food"].find({}))
    def score(food):
        s = 0
        tags = set(food.get("tags", []) or [])
        name = (food.get("name", "") or "").lower()
        # liked keywords in name or tags
        for l in liked:
            if l.lower() in name or l.lower() in tags:
                s += 3
        for d in disliked:
            if d.lower() in name or d.lower() in tags:
                s -= 5
        # condition suitability
        s += len(conditions.intersection(set(food.get("suitable_for", []) or []))) * 2
        # light preference for balanced macros if history high in one macro
        last3 = db["consumption"].find({"user_id": user_id}).sort("consumed_at", -1).limit(3)
        avg_protein = avg_carbs = avg_fat = 0
        cnt = 0
        for c in last3:
            n = c.get("nutrients", {})
            avg_protein += float(n.get("protein", 0))
            avg_carbs += float(n.get("carbs", 0))
            avg_fat += float(n.get("fat", 0))
            cnt += 1
        if cnt:
            avg_protein/=cnt; avg_carbs/=cnt; avg_fat/=cnt
            f_n = food.get("nutrients", {})
            # nudge towards protein if recent protein low
            if avg_protein < 25 and float(f_n.get("protein", 0)) >= 20:
                s += 1
            if avg_carbs > 150 and float(f_n.get("carbs", 0)) < 30:
                s += 1
        return s

    foods_sorted = sorted(foods, key=score, reverse=True)
    recs = [doc_to_public(f) for f in foods_sorted[:limit]]
    return {"items": recs}


@app.get("/recommendations/{user_id}/search")
def search_recommendations(user_id: str, q: str, conditions: Optional[str] = None, limit: int = 10):
    # premium-only check
    user = db["user"].find_one({"_id": ObjectId(user_id)}) if ObjectId.is_valid(user_id) else db["user"].find_one({"_id": ObjectId("0"*24)})
    u = db["user"].find_one({"_id": ObjectId(user_id)}) if ObjectId.is_valid(user_id) else None
    if not u or not u.get("is_premium"):
        raise HTTPException(status_code=403, detail="Premium only")
    filt: Dict[str, Any] = {"name": {"$regex": q, "$options": "i"}}
    if conditions:
        conds = [c.strip() for c in conditions.split(",") if c.strip()]
        filt["suitable_for"] = {"$in": conds}
    docs = db["food"].find(filt).limit(limit)
    return [doc_to_public(d) for d in docs]


# -------- Chatbot (simple heuristic) --------
class ChatMessage(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
def chat(msg: ChatMessage):
    prof = db["profile"].find_one({"user_id": msg.user_id}) or {}
    conds = set(prof.get("conditions", []) or [])
    advice = []
    text = msg.message.lower()
    if any(k in text for k in ["protein", "muscle", "gain"]):
        advice.append("Aim for 1.6-2.2 g/kg protein and include lean sources like chicken, tofu, or Greek yogurt.")
    if "diabet" in text or "sugar" in text or ("diabetes" in conds):
        advice.append("Prefer low-GI carbs, add fiber, and balance meals with protein and healthy fat.")
    if "hypertension" in text or "blood pressure" in text or ("hypertension" in conds):
        advice.append("Reduce sodium, favor potassium-rich foods (leafy greens, beans, bananas).")
    if not advice:
        advice.append("Focus on whole foods, adequate protein, colorful veggies, and consistent hydration.")
    # include quick suggestions
    recs = get_recommendations(msg.user_id)["items"] if db else []
    return {"reply": " ".join(advice), "suggestions": recs[:3]}


# -------- Planner & Grocery --------
class PlannerRequest(BaseModel):
    user_id: str
    week_of: datetime
    plan: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)


@app.post("/planner")
def save_planner(req: PlannerRequest):
    db["planner"].update_one({"user_id": req.user_id, "week_of": req.week_of}, {"$set": req.model_dump()}, upsert=True)
    return {"ok": True}


@app.get("/planner/{user_id}")
def get_planner(user_id: str, week_of: Optional[str] = None):
    filt = {"user_id": user_id}
    if week_of:
        try:
            filt["week_of"] = datetime.fromisoformat(week_of)
        except Exception:
            pass
    doc = db["planner"].find_one(filt)
    return doc_to_public(doc) if doc else {"user_id": user_id, "plan": {}}


class GroceryRequest(BaseModel):
    user_id: str
    food_ids: List[str]


@app.post("/grocery-list")
def grocery_list(req: GroceryRequest):
    ingredients: List[str] = []
    for fid in req.food_ids:
        try:
            f = db["food"].find_one({"_id": ObjectId(fid)})
            if f:
                ingredients.extend(f.get("ingredients", []))
        except Exception:
            continue
    # dedupe and normalize
    clean = []
    seen = set()
    for ing in ingredients:
        s = ing.strip().lower()
        if s not in seen:
            seen.add(s)
            clean.append(ing)
    gl = GroceryListSchema(user_id=req.user_id, name="Auto List", items=clean, generated_from=req.food_ids)
    gid = create_document("grocerylist", gl)
    return {"id": gid, "items": clean}


# -------- Misc & Health --------
@app.get("/")
def root():
    return {"message": "HLife API running"}


@app.get("/schema")
def get_schema():
    return {
        "collections": ["user", "profile", "food", "consumption", "weight", "planner", "grocerylist", "badge"],
    }


# -------- Seed sample foods on first run --------
@app.post("/seed")
def seed_foods():
    if db["food"].count_documents({}) > 0:
        return {"ok": True, "seeded": 0}
    samples = [
        {
            "name": "Grilled Chicken Salad",
            "description": "Lean protein with fresh veggies",
            "nutrients": {"calories": 350, "protein": 35, "carbs": 18, "fat": 14},
            "ingredients": ["chicken breast", "mixed greens", "olive oil", "tomatoes", "cucumber"],
            "recipe": ["Grill chicken", "Chop veggies", "Dress with olive oil"],
            "tags": ["lunch", "high-protein"],
            "suitable_for": ["diabetes", "weight_loss"],
        },
        {
            "name": "Oatmeal with Berries",
            "description": "Fiber-rich breakfast",
            "nutrients": {"calories": 280, "protein": 8, "carbs": 52, "fat": 5},
            "ingredients": ["rolled oats", "milk", "blueberries", "chia seeds"],
            "recipe": ["Cook oats in milk", "Top with berries and chia"],
            "tags": ["breakfast", "high-fiber"],
            "suitable_for": ["diabetes", "heart_health"],
        },
        {
            "name": "Tofu Stir-fry",
            "description": "Plant-based protein with veggies",
            "nutrients": {"calories": 420, "protein": 24, "carbs": 35, "fat": 20},
            "ingredients": ["tofu", "broccoli", "soy sauce", "garlic", "ginger"],
            "recipe": ["Sauté garlic and ginger", "Add tofu and broccoli", "Stir in soy sauce"],
            "tags": ["dinner", "vegan"],
            "suitable_for": ["hypertension", "vegetarian"],
        },
    ]
    for s in samples:
        create_document("food", FoodSchema(**s))
    return {"ok": True, "seeded": len(samples)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
