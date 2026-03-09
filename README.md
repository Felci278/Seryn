# Seryn: AI-Assisted Weekly Diet Planning for Type 2 Diabetic Emirati Patients

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

Seryn is an applied ML/AI project that explores a hybrid workflow for culturally aware meal planning:
1) profile understanding and data analysis,  
2) retrieval-based plan composition,  
4) structured weekly output.

The repository mixes **research notebooks** (experimentation) and **Python modules** (Gardio UI layer).

---

## 1. Problem Definition

Type 2 diabetes meal planning is constrained by multiple factors:

- Glycemic and nutrition constraints
- Individual preferences and adherence
- Cultural acceptability of foods
- Weekly-level planning consistency (not one-off meal suggestions)

Seryn targets this as a constrained recommendation/generation problem, with emphasis on Emirati dietary context.

---

## 2. Technical Scope

### Core objectives

- Build a repeatable pipeline for weekly diet plan generation.
- Support culturally relevant recommendations.
- Preserve machine-readable output for downstream UI/analysis.

---

## 3. Repository Layout (Technical)

```text
Seryn/
├── app_ui.py                
├── pipeline.py              
├── test.py                  
├── weekly_plan.json         
├── data_exploration.ipynb   
├── classification.ipynb     
├── generation.ipynb         
├── vector-search.ipynb      
├── README.md
├── LICENSE
└── .gitignore
```

---

## 4. System Architecture (Conceptual)

```text
[Input Profile / Constraints]
        |
        v
[Preprocessing + Feature Context]
        |
        +--> [Classification Module] ----+
        |                                |
        +--> [Vector Retrieval Module] --+--> [Plan Composer / Generator] --> [weekly_plan.json]
        |                                |
        +--> [Rule/Constraint Layer] ----+
                                         |
                                         v
                                     [UI Layer]
```

### Module responsibilities (high-level)

- **`pipeline.py`**
  - Orchestrates processing stages
  - Applies business logic/flow control
  - Produces serialized output artifacts

- **`app_ui.py`**
  - Handles user-facing interaction path
  - Bridges user inputs to pipeline execution
  - Presents or exports generated plans

- **Notebooks**
  - `data_exploration.ipynb`: feature distributions, data quality checks, exploratory findings
  - `classification.ipynb`: classifying the GI labels
  - `vector-search.ipynb`: similarity-based retrieval prototypes


---

## 5. Data & Output Contract

### Input (conceptual)
Expected input dimensions include:
- user demographics (optional based on implementation)
- diabetic-friendly constraints
- preference/cultural meal constraints
- planning horizon (weekly)

### Output
Primary output artifact:

- **`weekly_plan.json`**  
  A structured weekly representation suitable for:
  - UI rendering
  - reproducibility checks
  - potential API response payloads in future deployment


---

## 6. Execution Model

## 6.1 Environment Setup

```bash
git clone https://github.com/Felci278/Seryn.git
cd Seryn
python -m venv .venv
```

Activate environment:

- **macOS/Linux**
  ```bash
  source .venv/bin/activate
  ```

- **Windows (PowerShell)**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

Install dependencies (if available):
```bash
pip install -r requirements.txt
```

If no `requirements.txt` exists yet, install manually and freeze:
```bash
pip install <dependencies>
pip freeze > requirements.txt
```

## 6.2 Run Paths

### Pipeline execution
```bash
python pipeline.py
```

### UI path
```bash
python app_ui.py
```
---

## 7. License

MIT License — see [LICENSE](./LICENSE).
