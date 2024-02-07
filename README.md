# Impossible-Game

Repo for the "Integrationsseminar" at DHBW Mannheim

Erstellung eines Erklär-Videos mithilfe des packages Manim, in welchem das Thema Neuroevolution anhand einer Replikation der Videospiel-App "The impossible game", bzw. "Geometry Dash" erläutert wird.

Gruppenarbeit (3 Studierende; 2 oder 4 in Ausnahmen)

## Running the NEAT Algorithm

Tested on Ubuntu inside WSL-2 on Windows 11.

```python
conda env create --file 0__environment_run_neat.yml  # from src folder
conda activate neat
python 0_continuous_lunar_lander.py
```

- `0_continuous_lunar_lander.py` allows the modification of the different probabilities and other parameters of the NEAT algorithm (As well as specifying a different environment and fitness function).
- A run will create a folder inside `src/runs/` where the genotypes for each generation and their evaluated fitnesses are saved.
- Using `1_best_continuous_lunar_lander.py`, different landers can be visualized.
- The complete NEAT algorithm code is in `src/classes/NEAT.py`.

## Rendering the manim animation

Executed with pip venv on python 3.10.11.

```python
python -m venv .venv
.venv/Scripts/activate.bat
pip install -r src/5__requirements.txt
python 5_manim_main.py
```

- `5_manim_main.py` contains the code to render the manim animation. The script relies on classes and functions defined in `src/classes/manim_mobjects.py` and `src/classes/manim_utils.py`. The voiceover, music and lunar lander video clips are not part of the animation and have to be added in post production. The script uses the information inside `src/runs/` to visualize the actual results of the NEAT Algorithm.
- `6_record_gymnasium.py` contains the code to record a lunar lander clip. Each execution will create a folder inside `gymnasium_videos\<run>\`. These folders contain the video clip as well a json file in which the activation of each neuron of the neural network is documented for each frame.
