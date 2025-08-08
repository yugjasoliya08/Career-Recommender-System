import time
import os
from joblib import load
import numpy as np

# Third-party libraries for a better CLI experience
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import pyfiglet

# Initialize Rich Console for beautiful printing
console = Console()


def load_resources():
    """Loads and returns the model and encoders."""
    model = load(os.path.join("model", "model.pkl"))
    le_skills = load(os.path.join("model", "skills_encoder.pkl"))
    le_interest = load(os.path.join("model", "interest_encoder.pkl"))
    le_career = load(os.path.join("model", "career_encoder.pkl"))
    return model, le_skills, le_interest, le_career


def get_user_input(skills_options, interests_options):
    """
    Prompts the user for input using interactive menus.
    """
    console.print(Panel.fit("[bold cyan]👤 Please provide your details[/bold cyan]"))

    gpa = questionary.text(
        "What is your GPA (0.0 - 10.0)?",
        validate=lambda val: (
            True if 0.0 <= float(val) <= 10.0 else "Please enter a number between 0.0 and 10.0"
        )
    ).ask()

    # Using select for skills and interests prevents typos and validation issues
    skill = questionary.select(
        "What is your top skill?",
        choices=skills_options
    ).ask()

    interest = questionary.select(
        "What is your primary interest?",
        choices=interests_options
    ).ask()
    
    if gpa is None or skill is None or interest is None:
        console.print("[bold red]Input cancelled. Exiting.[/bold red]")
        exit()

    return float(gpa), skill, interest


def display_results(gpa, skill, interest, career):
    """Displays the prediction result in a formatted panel."""
    
    # Create a table for the user's input
    input_table = Table(show_header=False, box=None, padding=(0, 2))
    input_table.add_column(style="dim")
    input_table.add_column(style="bold")
    input_table.add_row("GPA", f"{gpa:.2f}")
    input_table.add_row("Top Skill", skill)
    input_table.add_row("Primary Interest", interest)

    # Create the main results panel
    results_panel = Panel(
        f"\n[bold green]🎯 Recommended Career:[/bold green]\n[bold magenta not dim]👉 {career}[/bold magenta not dim]\n",
        title="[bold]✨ Your Recommendation[/bold]",
        subtitle="[dim]Based on your profile[/dim]",
        border_style="green",
        expand=False,
        padding=(1, 2)
    )

    console.print("\nHere's what we analyzed:")
    console.print(input_table)
    console.print(results_panel)


def main():
    """Main function to run the application."""
    try:
        # --- Welcome Banner ---
        banner = pyfiglet.figlet_format("Career AI", font="slant")
        console.print(f"[bold green]{banner}[/bold green]")
        console.print(Panel.fit("Welcome to the AI Career Recommender System!", border_style="blue"))

        # --- Load Resources ---
        model, le_skills, le_interest, le_career = load_resources()
        skills_options = sorted(le_skills.classes_)
        interests_options = sorted(le_interest.classes_)

        # --- Get User Input ---
        gpa, skill, interest = get_user_input(skills_options, interests_options)

        # --- Predict ---
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Analyzing your profile...", total=None)
            skill_encoded = le_skills.transform([skill])[0]
            interest_encoded = le_interest.transform([interest])[0]
            features = np.array([[gpa, skill_encoded, interest_encoded]])
            prediction_encoded = model.predict(features)
            recommended_career = le_career.inverse_transform(prediction_encoded)[0]
            time.sleep(1) # Simulate processing time

        # --- Display Results ---
        display_results(gpa, skill, interest, recommended_career)

    except (KeyboardInterrupt, TypeError):
        console.print("\n[bold red]✖️ Program interrupted by user. Exiting.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")


if __name__ == "__main__":
    main()