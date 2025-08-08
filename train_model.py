import pandas as pd
import os
from joblib import dump

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Rich library for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize Rich Console
console = Console()

def main():
    """Main function to orchestrate the model training process."""
    console.print(Panel.fit("[bold cyan]🚀 Starting AI Model Training and Evaluation 🚀[/bold cyan]", border_style="blue"))
    try:
        console.print("\n[yellow]Step 1: Loading Dataset...[/yellow]")
        df = pd.read_csv("dataset/career_data.csv")
        console.print(f"✅ Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

        console.print("\n[yellow]Step 2: Preprocessing Data...[/yellow]")
        
        # FINAL FIX: Clean all categorical columns, including the target career
        for col in ["Skills", "Interest", "Recommended_Career"]:
            df[col] = df[col].str.strip().str.title()
        console.print("✅ Data cleaning (trimming whitespace, standardizing case) complete.")

        le_skills = LabelEncoder()
        le_interest = LabelEncoder()
        le_career = LabelEncoder()

        df["Skills"] = le_skills.fit_transform(df["Skills"])
        df["Interest"] = le_interest.fit_transform(df["Interest"])
        df["Recommended_Career"] = le_career.fit_transform(df["Recommended_Career"])
        console.print("✅ Categorical features encoded successfully.")

        console.print("\n[yellow]Step 3: Splitting Data into Training and Testing Sets...[/yellow]")
        X = df[["GPA", "Skills", "Interest"]]
        y = df["Recommended_Career"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        console.print(f"✅ Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

        console.print("\n[yellow]Step 4: Training the Model...[/yellow]")
        model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
        model.fit(X_train, y_train)
        console.print("✅ Model training complete using RandomForestClassifier.")

        console.print("\n[yellow]Step 5: Evaluating Model Performance...[/yellow]")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        report_table = Table(show_header=True, header_style="bold magenta", title="Model Performance on Test Data")
        report_table.add_column("Metric", style="cyan")
        report_table.add_column("Value", style="green")
        report_table.add_row("Overall Accuracy", f"{accuracy:.2%}")
        report_table.add_row("Out-of-Bag (OOB) Score", f"{model.oob_score_:.2%}")
        
        console.print(report_table)
        
        report = classification_report(y_test, y_pred, target_names=le_career.classes_, zero_division=0)
        console.print(Panel(report, title="[bold]Classification Report[/bold]", border_style="cyan", expand=False))

        console.print("\n[yellow]Step 6: Saving Model and Encoders...[/yellow]")
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)

        dump(model, os.path.join(model_dir, "model.pkl"))
        dump(le_skills, os.path.join(model_dir, "skills_encoder.pkl"))
        dump(le_interest, os.path.join(model_dir, "interest_encoder.pkl"))
        dump(le_career, os.path.join(model_dir, "career_encoder.pkl"))
        console.print(f"✅ Model and encoders successfully saved to the '{model_dir}/' directory.")
        
        console.print(Panel.fit("[bold green]🎉 Training process completed successfully! 🎉[/bold green]", border_style="green"))

    except FileNotFoundError:
        console.print("[bold red]Error: 'dataset/career_data.csv' not found. Please ensure the dataset is in the correct directory.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")

if __name__ == "__main__":
    main()