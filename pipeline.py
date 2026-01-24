import json
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

# 1. DATA MODELS & SCHEMAS
# =====================================

@dataclass
class PatientProfile:
    """Patient profile input schema"""
    age: int
    sex: str
    bmi: float
    diabetes_type: str
    hba1c: float
    calorie_needs: int
    carb_tolerance: int
    sugar_sensitivity: str
    activity_level: str
    dietary_restrictions: Optional[List[str]] = None
    cultural_preferences: Optional[str] = "Emirati"

class DailyMeal(BaseModel):
    """Single day meal plan schema"""
    breakfast: str = Field(description="Breakfast dish with portion size")
    lunch: str = Field(description="Lunch dish with portion size")
    dinner: str = Field(description="Dinner dish with portion size")
    snack: str = Field(description="Healthy snack option")
    drink: str = Field(description="Recommended beverage")

class WeeklyMealPlan(BaseModel):
    """7-day meal plan schema"""
    day1: DailyMeal
    day2: DailyMeal
    day3: DailyMeal
    day4: DailyMeal
    day5: DailyMeal
    day6: DailyMeal
    day7: DailyMeal
    nutritional_notes: str = Field(description="Overall nutritional guidance")
    total_estimated_calories: str = Field(description="Daily calorie estimate range")

# =====================================
# 2. FOOD SAFETY CLASSIFIER
# =====================================

class FoodSafetyClassifier:
    """Rule-based safety classifier for diabetic patients"""
    
    def __init__(self):
        self.high_risk_keywords = [
            "sugar", "syrup", "honey", "candy", "cake", "pastry", "donut",
            "soda", "juice", "sweetened", "fried", "deep-fried"
        ]
        self.medium_risk_keywords = [
            "rice", "bread", "pasta", "potato", "banana", "mango", "dates"
        ]
        
    def classify_food(self, food_name: str, gi_label: str = None) -> str:
        """
        Classify food safety for diabetic patients
        Returns: 'safe', 'limited', 'avoid'
        """
        food_lower = food_name.lower()
        
        # Use GI label if available
        if gi_label:
            if gi_label.lower() == 'friendly':
                return 'safe'
            elif gi_label.lower() == 'limited':
                return 'limited'
            elif gi_label.lower() == 'avoid':
                return 'avoid'
        
        # Rule-based classification
        if any(keyword in food_lower for keyword in self.high_risk_keywords):
            return 'avoid'
        elif any(keyword in food_lower for keyword in self.medium_risk_keywords):
            return 'limited'
        else:
            return 'safe'

# =====================================
# 3. CHROMADB VECTOR STORE WRAPPER
# =====================================

class DiabeticFoodVectorStore:
    """ChromaDB wrapper for food recommendations"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.classifier = FoodSafetyClassifier()
        
        # Load collections
        try:
            self.food_collection = self.client.get_collection("food_items")
            self.patient_collection = self.client.get_collection("patient_profiles")
            self.meal_collections = {}
            for meal_type in ["breakfast", "lunch", "dinner", "snacks", "drinks"]:
                try:
                    self.meal_collections[meal_type] = self.client.get_collection(f"{meal_type}_foods")
                except:
                    print(f"Warning: {meal_type}_foods collection not found")
        except Exception as e:
            print(f"Warning: Could not load ChromaDB collections: {e}")
    
    def get_safe_foods_by_meal(self, meal_type: str, patient_profile: PatientProfile, n_results: int = 20) -> List[str]:
        """Retrieve safe foods for specific meal type based on patient profile"""
        
        # Create search query based on patient needs
        query_text = self._create_patient_query(patient_profile, meal_type)
        
        # Try meal-specific collection first
        if meal_type in self.meal_collections:
            try:
                results = self.meal_collections[meal_type].query(
                    query_texts=[query_text],
                    n_results=n_results
                )
                foods = results['documents'][0] if results['documents'] else []
            except:
                foods = []
        else:
            # Fallback to general food collection
            try:
                results = self.food_collection.query(
                    query_texts=[query_text + f" {meal_type}"],
                    n_results=n_results
                )
                foods = results['documents'][0] if results['documents'] else []
            except:
                foods = []
        
        # Filter by safety classification
        safe_foods = []
        for food in foods:
            safety_class = self.classifier.classify_food(food)
            if safety_class in ['safe', 'limited']:
                safe_foods.append(food)
        
        return safe_foods[:15]  # Limit to top 15 safe options
    
    def _create_patient_query(self, profile: PatientProfile, meal_type: str) -> str:
        """Create search query based on patient profile"""
        sensitivity_map = {"high": "very low sugar", "medium": "moderate sugar", "low": "low sugar"}
        activity_map = {"high": "high protein", "moderate": "balanced", "low": "light"}
        
        query = f"""
        {meal_type} for {profile.diabetes_type} diabetes patient
        {sensitivity_map.get(profile.sugar_sensitivity.lower(), "moderate sugar")}
        {activity_map.get(profile.activity_level.lower(), "balanced")} meal
        {profile.cultural_preferences or "traditional"} cuisine
        healthy diabetic-friendly low glycemic
        """
        
        return query.strip()

# =====================================
# 4. LANGCHAIN TOOLS
# =====================================

def create_food_retrieval_tool(vector_store: DiabeticFoodVectorStore) -> Tool:
    """Create tool for retrieving safe foods"""
    
    def retrieve_foods(query: str) -> str:
        """Retrieve safe foods based on meal type and patient needs"""
        try:
            # Parse query to extract meal type and patient info
            parts = query.split("|")
            meal_type = parts[0].strip() if len(parts) > 0 else "breakfast"
            
            # Create dummy patient profile for demo (in real use, this would come from context)
            profile = PatientProfile(
                age=50, sex="female", bmi=25.0, diabetes_type="Type 2",
                hba1c=7.0, calorie_needs=1800, carb_tolerance=150,
                sugar_sensitivity="medium", activity_level="moderate"
            )
            
            foods = vector_store.get_safe_foods_by_meal(meal_type, profile, n_results=15)
            return f"Safe {meal_type} options: {', '.join(foods[:10])}"
        except Exception as e:
            return f"Error retrieving foods: {str(e)}"
    
    return Tool(
        name="food_retrieval",
        description="Retrieve safe diabetic-friendly foods for specific meal types. Input format: 'meal_type|patient_context'",
        func=retrieve_foods
    )

def create_nutrition_analysis_tool() -> Tool:
    """Create tool for nutritional analysis"""
    
    def analyze_nutrition(meal_description: str) -> str:
        """Analyze nutritional content of meal"""
        # This would integrate with a nutrition API in production
        # For now, providing rule-based analysis
        
        keywords = meal_description.lower()
        analysis = []
        
        if any(term in keywords for term in ["salad", "vegetable", "green"]):
            analysis.append("High in fiber and micronutrients")
        if any(term in keywords for term in ["fish", "chicken", "lean"]):
            analysis.append("Good protein source")
        if any(term in keywords for term in ["whole grain", "brown", "oat"]):
            analysis.append("Complex carbohydrates")
        if any(term in keywords for term in ["fried", "sugar", "sweet"]):
            analysis.append("⚠️ May spike blood glucose")
            
        return f"Nutritional analysis: {'; '.join(analysis) if analysis else 'Balanced meal option'}"
    
    return Tool(
        name="nutrition_analysis",
        description="Analyze nutritional content and diabetic suitability of meals",
        func=analyze_nutrition
    )

# =====================================
# 5. MEAL PLAN GENERATOR CHAIN
# =====================================

class MealPlanGenerator:
    """LangChain-based meal plan generator"""
    
    def __init__(self, llm, vector_store: DiabeticFoodVectorStore):
        self.llm = llm
        self.vector_store = vector_store
        self.parser = PydanticOutputParser(pydantic_object=WeeklyMealPlan)
        
        # Create the main prompt template
        self.prompt_template = PromptTemplate(
            input_variables=[
                "patient_profile", "safe_breakfast", "safe_lunch", 
                "safe_dinner", "safe_snacks", "safe_drinks"
            ],
            template="""
You are an expert clinical dietitian specializing in diabetes management. Create a personalized 7-day meal plan.

PATIENT PROFILE:
{patient_profile}

AVAILABLE SAFE FOODS:
- Breakfast options: {safe_breakfast}
- Lunch options: {safe_lunch}  
- Dinner options: {safe_dinner}
- Snack options: {safe_snacks}
- Drink options: {safe_drinks}

REQUIREMENTS:
1. Use ONLY foods from the provided safe lists
2. Ensure variety across the week (no food repeated more than twice)
3. Balance macronutrients appropriately for diabetes management
4. Include portion sizes (e.g., "1 cup", "small portion")
5. Consider cultural preferences and meal timing
6. Aim for glycemic load balance throughout each day

{format_instructions}

Generate a complete 7-day meal plan that follows diabetic dietary guidelines while being culturally appropriate and varied.

Respond ONLY with a valid JSON object matching the schema above. Do not include any explanation or extra text.
            """,
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        # Create the chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            output_parser=self.parser
        )
    
    def generate_meal_plan(self, patient_profile: PatientProfile) -> WeeklyMealPlan:
        """Generate complete weekly meal plan"""
        
        # Retrieve safe foods for each meal type
        safe_foods = {}
        meal_types = ["breakfast", "lunch", "dinner", "snacks", "drinks"]
        
        if self.vector_store:
            for meal_type in meal_types:
                try:
                    foods = self.vector_store.get_safe_foods_by_meal(meal_type, patient_profile, n_results=15)
                    safe_foods[f"safe_{meal_type}"] = ", ".join(foods) if foods else "oat porridge, grilled fish, vegetables, nuts, water"
                except Exception as e:
                    print(f"Error retrieving {meal_type} foods: {e}")
                    safe_foods[f"safe_{meal_type}"] = "basic healthy options available"
        else:
            # Fallback foods when vector store is not available
            fallback_foods = {
                "safe_breakfast": "oat porridge, whole wheat toast, egg scramble, yogurt, herbal tea",
                "safe_lunch": "grilled fish, lentil salad, quinoa bowl, vegetable soup, green tea", 
                "safe_dinner": "grilled chicken, steamed vegetables, barley soup, mixed salad, water",
                "safe_snacks": "mixed nuts, low-fat yogurt, apple slices, cucumber, herbal tea",
                "safe_drinks": "water, herbal tea, green tea, lemon water, unsweetened beverages"
            }
            safe_foods = fallback_foods
        
        # Create patient profile string
        profile_str = f"""
        Age: {patient_profile.age}, Sex: {patient_profile.sex}
        Diabetes Type: {patient_profile.diabetes_type}
        HbA1c: {patient_profile.hba1c}%
        Daily Calories: {patient_profile.calorie_needs} kcal
        Carb Tolerance: {patient_profile.carb_tolerance}g
        Sugar Sensitivity: {patient_profile.sugar_sensitivity}
        Activity Level: {patient_profile.activity_level}
        Cultural Preference: {patient_profile.cultural_preferences}
        """
        
        # Generate the meal plan
        try:
            result = self.chain.run(
                patient_profile=profile_str.strip(),
                **safe_foods
            )
            return result
        except Exception as e:
            print(f"Error generating meal plan: {e}")
            # Return structured fallback plan that matches WeeklyMealPlan schema
            fallback_meal = DailyMeal(
                breakfast="Oat porridge with berries (1 cup)",
                lunch="Grilled fish with vegetables (1 portion)",
                dinner="Lentil soup with salad (1 bowl + side)",
                snack="Mixed nuts (small handful)",
                drink="Water or herbal tea"
            )
            
            # Create complete 7-day fallback plan
            return WeeklyMealPlan(
                day1=fallback_meal,
                day2=DailyMeal(
                    breakfast="Whole wheat toast with avocado",
                    lunch="Chickpea salad with vegetables", 
                    dinner="Grilled chicken with steamed broccoli",
                    snack="Low-fat yogurt",
                    drink="Green tea"
                ),
                day3=DailyMeal(
                    breakfast="Egg scramble with spinach",
                    lunch="Quinoa bowl with vegetables",
                    dinner="Baked fish with roasted vegetables", 
                    snack="Apple slices with almond butter",
                    drink="Herbal tea"
                ),
                day4=fallback_meal,
                day5=fallback_meal, 
                day6=fallback_meal,
                day7=fallback_meal,
                nutritional_notes="Basic diabetic-friendly meal plan. Please consult with a registered dietitian for personalized recommendations.",
                total_estimated_calories="1600-1800 kcal/day"
            )

# =====================================
# 6. MAIN SYSTEM ORCHESTRATOR
# =====================================

class DiabeticMealPlanSystem:
    """Main system orchestrating all components"""
    
    def __init__(self, anthropic_api_key: str = None, chroma_db_path: str = "./chroma_db"):
        # Initialize LLM (using Claude via Anthropic)
        if anthropic_api_key:
            self.llm = ChatAnthropic(
                anthropic_api_key="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",
                model="claude-3-haiku-20240307",
                temperature=0.3
            )
        else:
            # Fallback to a mock LLM for demo purposes
            from langchain.llms.fake import FakeListLLM
            self.llm = FakeListLLM(responses=[
                json.dumps({
                    "day1": {"breakfast": "Oat porridge (1 cup)", "lunch": "Grilled fish (1 portion)", 
                           "dinner": "Vegetable soup (1 bowl)", "snack": "Nuts (handful)", "drink": "Water"},
                    "day2": {"breakfast": "Egg wrap (1 small)", "lunch": "Lentil salad (1 cup)",
                           "dinner": "Grilled chicken (1 portion)", "snack": "Yogurt (small)", "drink": "Tea"},
                    "day3": {"breakfast": "Chickpea flour flatbread", "lunch": "Vegetable curry", 
                           "dinner": "Grilled fish with salad", "snack": "Low-fat yogurt", "drink": "Herbal tea"},
                    "day4": {"breakfast": "Whole wheat toast", "lunch": "Lentil soup", 
                           "dinner": "Grilled chicken breast", "snack": "Mixed nuts", "drink": "Water"},
                    "day5": {"breakfast": "Oat porridge with berries", "lunch": "Quinoa salad", 
                           "dinner": "Steamed vegetables", "snack": "Fresh fruit", "drink": "Green tea"},
                    "day6": {"breakfast": "Egg scramble", "lunch": "Fish curry", 
                           "dinner": "Roasted vegetables", "snack": "Almonds", "drink": "Lemon water"},
                    "day7": {"breakfast": "Whole grain cereal", "lunch": "Bean salad", 
                           "dinner": "Grilled lean meat", "snack": "Cucumber slices", "drink": "Mint tea"},
                    "nutritional_notes": "Balanced diabetic meal plan with cultural foods, portion-controlled for blood sugar management",
                    "total_estimated_calories": "1600-1800 kcal/day"
                })
            ])
        
        # Initialize vector store
        try:
            self.vector_store = DiabeticFoodVectorStore(chroma_db_path)
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            self.vector_store = None
        
        # Initialize meal plan generator
        self.meal_generator = MealPlanGenerator(self.llm, self.vector_store)
        
        # Create tools
        self.tools = [
            create_food_retrieval_tool(self.vector_store) if self.vector_store else None,
            create_nutrition_analysis_tool()
        ]
        # Filter out None tools
        self.tools = [tool for tool in self.tools if tool is not None]
    
    def generate_personalized_meal_plan(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for generating meal plans"""
        
        try:
            # 1. Parse patient profile
            profile = PatientProfile(**patient_data)
            
            # 2. Generate meal plan using LangChain
            meal_plan = self.meal_generator.generate_meal_plan(profile)
            
            # 3. Convert to dictionary for JSON output
            return meal_plan.dict()
            
        except Exception as e:
            print(f"Error in meal plan generation: {e}")
            return {
                "error": str(e),
                "fallback_plan": "Please consult with a dietitian for personalized meal planning"
            }

# =====================================
# 7. EXAMPLE USAGE & TESTING
# =====================================

def demo_system():
    """Demonstrate the meal planning system"""
    
    # Initialize system (without API key for demo)
    system = DiabeticMealPlanSystem("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")
    
    # Example patient profile
    patient_data = {
        "age": 58,
        "sex": "female",
        "bmi": 27.5,
        "diabetes_type": "Type 2",
        "hba1c": 7.3,
        "calorie_needs": 1660,
        "carb_tolerance": 131,
        "sugar_sensitivity": "High",
        "activity_level": "Moderate",
        "cultural_preferences": "Emirati"
    }
    
    # Generate meal plan
    print("Generating personalized meal plan...")
    meal_plan = system.generate_personalized_meal_plan(patient_data)
    
    # Display results
    print("\nGenerated Meal Plan:")
    print(json.dumps(meal_plan, indent=2))
    
    # Test individual tools
    print("\nTesting food retrieval tool:")
    food_tool = system.tools[0]
    breakfast_foods = food_tool.func("breakfast|diabetic patient")
    print(breakfast_foods)
    
    print("\nTesting nutrition analysis tool:")
    nutrition_tool = system.tools[1]
    analysis = nutrition_tool.func("grilled fish with vegetables and brown rice")
    print(analysis)

if __name__ == "__main__":
    demo_system()