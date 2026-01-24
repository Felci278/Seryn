import gradio as gr
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import traceback
import os
from datetime import datetime



from pipeline import (
    PatientProfile, WeeklyMealPlan, DiabeticMealPlanSystem,
    FoodSafetyClassifier, DiabeticFoodVectorStore
)

# =====================================
# GRADIO INTERFACE COMPONENTS
# =====================================

class GradioDietInterface:
    """Gradio interface wrapper for the diet planning system"""
    
    def __init__(self, anthropic_api_key: str = None, chroma_db_path: str = "./chroma_db"):
        try:
            self.diet_system = DiabeticMealPlanSystem(anthropic_api_key, chroma_db_path)
            self.system_ready = True
            self.status_message = "System initialized successfully"
        except Exception as e:
            self.system_ready = False
            self.status_message = f"System initialization error: {str(e)}"
            print(f"Warning: {self.status_message}")
    
    def generate_meal_plan_interface(
        self,
        age: int,
        sex: str,
        weight: float,
        height: float,
        diabetes_type: str,
        hba1c: float,
        activity_level: str,
        sugar_sensitivity: str,
        dietary_restrictions: str,
        cultural_preferences: str,
        target_calories: int,
        carb_limit: int
    ) -> Tuple[str, str, str]:
        """Generate meal plan from Gradio inputs"""
        
        try:
            # Validate inputs
            if not self.system_ready:
                return (
                    "❌ System Error", 
                    self.status_message, 
                    "Please check your ChromaDB setup and API configuration."
                )
            
            # Calculate BMI
            height_m = height / 100  # Convert cm to meters
            bmi = weight / (height_m ** 2)
            
            # Parse dietary restrictions
            restrictions = []
            if dietary_restrictions.strip():
                restrictions = [r.strip() for r in dietary_restrictions.split(",")]
            
            # Create patient profile
            patient_data = {
                "age": age,
                "sex": sex.lower(),
                "bmi": round(bmi, 1),
                "diabetes_type": diabetes_type,
                "hba1c": hba1c,
                "calorie_needs": target_calories,
                "carb_tolerance": carb_limit,
                "sugar_sensitivity": sugar_sensitivity,
                "activity_level": activity_level,
                "dietary_restrictions": restrictions if restrictions else None,
                "cultural_preferences": cultural_preferences
            }
            
            # Generate meal plan
            meal_plan_result = self.diet_system.generate_personalized_meal_plan(patient_data)
            
            # Format results for display
            if "error" in meal_plan_result:
                return (
                    "⚠️ Generation Error",
                    meal_plan_result.get("error", "Unknown error"),
                    meal_plan_result.get("fallback_plan", "Please try again or consult a dietitian")
                )
            
            # Create formatted meal plan display
            meal_plan_text = self._format_meal_plan_display(meal_plan_result, patient_data)
            
            # Create summary
            summary = self._create_patient_summary(patient_data, bmi)
            
            # Create JSON output for download
            json_output = json.dumps(meal_plan_result, indent=2, ensure_ascii=False)
            
            return (
                "✅ Meal Plan Generated Successfully",
                summary,
                meal_plan_text
            )
            
        except Exception as e:
            error_msg = f"Error generating meal plan: {str(e)}"
            return (
                "❌ Error",
                error_msg,
                f"Technical details: {traceback.format_exc()}"
            )
    
    def _format_meal_plan_display(self, meal_plan: Dict, patient_data: Dict) -> str:
        """Format meal plan for readable display"""
        
        try:
            output = []
            output.append("# Patient's \n")
            
            # Patient info header
            output.append(f"**Patient:** {patient_data['age']} year old {patient_data['sex']}")
            output.append(f"**Diabetes Type:** {patient_data['diabetes_type']}")
            output.append(f"**Target Calories:** {patient_data['calorie_needs']} kcal/day")
            output.append(f"**Carb Limit:** {patient_data['carb_tolerance']}g/day\n")
            
            # Daily meal plans
            for day_num in range(1, 8):
                day_key = f"day{day_num}"
                if day_key in meal_plan:
                    day_data = meal_plan[day_key]
                    
                    output.append(f"## 📅 Day {day_num}")
                    output.append(f"**🌅 Breakfast:** {day_data.get('breakfast', 'Not specified')}")
                    output.append(f"**🍽️ Lunch:** {day_data.get('lunch', 'Not specified')}")
                    output.append(f"**🌆 Dinner:** {day_data.get('dinner', 'Not specified')}")
                    output.append(f"**🥨 Snack:** {day_data.get('snack', 'Not specified')}")
                    output.append(f"**🥤 Drink:** {day_data.get('drink', 'Not specified')}")
                    output.append("")
            
            # Nutritional notes
            if "nutritional_notes" in meal_plan:
                output.append("## 📋 Nutritional Guidelines")
                output.append(meal_plan["nutritional_notes"])
                output.append("")
            
            if "total_estimated_calories" in meal_plan:
                output.append(f"**📊 Daily Calorie Range:** {meal_plan['total_estimated_calories']}")
            
            # Add disclaimer
            output.append("\n---")
            output.append("⚠️ **Medical Disclaimer:** This meal plan is for informational purposes only. Please rely on your judgement ")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error formatting meal plan: {str(e)}"
    
    def _create_patient_summary(self, patient_data: Dict, bmi: float) -> str:
        """Create patient profile summary"""
        
        # BMI classification
        if bmi < 18.5:
            bmi_class = "Underweight"
        elif bmi < 25:
            bmi_class = "Normal weight"
        elif bmi < 30:
            bmi_class = "Overweight"
        else:
            bmi_class = "Obese"
        
        # HbA1c assessment
        hba1c = patient_data["hba1c"]
        if hba1c < 7:
            hba1c_status = "Good control"
        elif hba1c < 8:
            hba1c_status = "Fair control"
        else:
            hba1c_status = "Poor control - needs attention"
        
        summary = f"""
## 👤 Patient Profile Summary

- **Age:** {patient_data['age']} years
- **BMI:** {bmi:.1f} ({bmi_class})
- **Diabetes Control:** HbA1c {hba1c}% ({hba1c_status})
- **Activity Level:** {patient_data['activity_level']}
- **Sugar Sensitivity:** {patient_data['sugar_sensitivity']}
- **Cultural Preference:** {patient_data['cultural_preferences']}
        """
        
        return summary.strip()
    
    def get_food_suggestions(self, meal_type: str, restrictions: str) -> str:
        """Get food suggestions for specific meal type"""
        try:
            if not self.system_ready:
                return "System not ready. Please check ChromaDB setup."
            
            # Use food retrieval tool
            food_tool = self.diet_system.tools[0]
            query = f"{meal_type.lower()}|{restrictions}"
            suggestions = food_tool.func(query)
            
            return f"### {meal_type} Suggestions\n{suggestions}"
            
        except Exception as e:
            return f"Error getting suggestions: {str(e)}"

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize the diet interface
    interface = GradioDietInterface("test_api_key")
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .output-markdown {
        font-size: 14px;
        line-height: 1.6;
    }
    .input-group {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Diabetic Meal Plan Generator") as app:
        
        # Header
        gr.Markdown("""
        # 🍽️ سىرِن  SERYN
        
        Generate personalized, culturally-appropriate meal plans for diabetes management using advanced AI and nutritional science.
        
        **Features:**
        - Personalized 7-day meal plans
        - Diabetic-friendly food recommendations
        - Cultural cuisine preferences
        - Nutritional analysis and safety filtering
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Patient Information")
                
                # Basic demographics
                with gr.Group():
                    gr.Markdown("### Personal Details")
                    age = gr.Slider(
                        minimum=18, maximum=100, value=50, step=1,
                        label="Age (years)"
                    )
                    sex = gr.Dropdown(
                        choices=["Male", "Female"], value="Female",
                        label="Sex"
                    )
                    
                # Physical measurements
                with gr.Group():
                    gr.Markdown("### Physical Measurements")
                    weight = gr.Number(
                        value=70, minimum=30, maximum=200,
                        label="Weight (kg)"
                    )
                    height = gr.Number(
                        value=165, minimum=120, maximum=220,
                        label="Height (cm)"
                    )
                
                # Medical information
                with gr.Group():
                    gr.Markdown("### Medical Information")
                    diabetes_type = gr.Dropdown(
                        choices=["Type 1", "Type 2", "Gestational"], 
                        value="Type 2",
                        label="Diabetes Type"
                    )
                    hba1c = gr.Number(
                        value=7.0, minimum=4.0, maximum=15.0, step=0.1,
                        label="HbA1c (%)"
                    )
                
                # Lifestyle factors
                with gr.Group():
                    gr.Markdown("### Lifestyle Factors")
                    activity_level = gr.Dropdown(
                        choices=["Low", "Moderate", "High"], 
                        value="Moderate",
                        label="Activity Level"
                    )
                    sugar_sensitivity = gr.Dropdown(
                        choices=["Low", "Medium", "High"], 
                        value="Medium",
                        label="Sugar Sensitivity"
                    )
                
                # Dietary preferences
                with gr.Group():
                    gr.Markdown("### Dietary Preferences")
                    dietary_restrictions = gr.Textbox(
                        label="Dietary Restrictions",
                        placeholder="e.g., gluten-free, dairy-free, vegetarian (comma-separated)",
                        lines=2
                    )
                    cultural_preferences = gr.Dropdown(
                        choices=["Emirati", "Arabic", "Mediterranean",], 
                        value="Emirati",
                        label="Cultural Cuisine Preference"
                    )
                
                # Nutritional targets
                with gr.Group():
                    gr.Markdown("### Nutritional Targets")
                    target_calories = gr.Slider(
                        minimum=1200, maximum=3000, value=1800, step=50,
                        label="Target Daily Calories"
                    )
                    carb_limit = gr.Slider(
                        minimum=50, maximum=300, value=150, step=10,
                        label="Daily Carbohydrate Limit (g)"
                    )
                
                # Generate button
                generate_btn = gr.Button(
                    "🔄 Generate Meal Plan", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Results")
                
                # Status output
                status_output = gr.Textbox(
                    label="Generation Status",
                    interactive=False
                )
                
                # Patient summary
                summary_output = gr.Markdown(
                    label="Patient Summary"
                )
                
                # Main meal plan output
                meal_plan_output = gr.Markdown(
                    label="7-Day Meal Plan",
                    show_label=False
                )
                
                # Additional tools section
                with gr.Accordion("🔍 Additional Tools", open=False):
                    gr.Markdown("### Food Suggestion Tool")
                    with gr.Row():
                        meal_type_input = gr.Dropdown(
                            choices=["Breakfast", "Lunch", "Dinner", "Snacks", "Drinks"],
                            value="Breakfast",
                            label="Meal Type"
                        )
                        restrictions_input = gr.Textbox(
                            label="Dietary Context",
                            placeholder="e.g., low sugar, high protein"
                        )
                    
                    suggest_btn = gr.Button("Get Suggestions")
                    suggestions_output = gr.Markdown()
        
        # Event handlers
        generate_btn.click(
            fn=interface.generate_meal_plan_interface,
            inputs=[
                age, sex, weight, height, diabetes_type, hba1c,
                activity_level, sugar_sensitivity, dietary_restrictions,
                cultural_preferences, target_calories, carb_limit
            ],
            outputs=[status_output, summary_output, meal_plan_output]
        )
        
        suggest_btn.click(
            fn=interface.get_food_suggestions,
            inputs=[meal_type_input, restrictions_input],
            outputs=[suggestions_output]
        )
        
        # Footer with instructions
        gr.Markdown("""
        ---
        ### 📌 Usage Instructions:
        1. **Fill in pateint's personal and medical information** in the left panel
        2. **Set patient's dietary preferences** and nutritional targets  
        3. **Click "Generate Meal Plan"** to create the personalized 7-day plan
        4. **Use additional tools** to explore food suggestions for specific meals
        
        ### ⚠️ Important Notes:
        - This tool is for educational purposes and meal planning assistance
        - Always rely on your own judgement before making dietary changes
        - The AI recommendations should complement, not replace, professional medical advice
        - Monitor patient's blood glucose levels when trying new foods
        
        ### 🔧 System Requirements:
        - ChromaDB database with food embeddings
        - Anthropic API key (optional, uses fallback otherwise)
        - Internet connection for AI model access
        """)
    
    return app

# =====================================
# MAIN EXECUTION
# =====================================

def main():
    """Main function to launch the Gradio interface"""
    
    # Create and launch the interface
    app = create_gradio_interface()
    
    # Launch with custom settings
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True,             # Enable debug mode
        show_error=True,        # Show detailed errors
        inbrowser=True          # Auto-open browser
    )

if __name__ == "__main__":
    # Add requirements check
    try:
        import gradio
        print("✅ Gradio imported successfully")
    except ImportError:
        print("❌ Gradio not found. Install with: pip install gradio")
        exit(1)
    
    main()