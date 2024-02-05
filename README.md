# Impossible-Game

Repo for the "Integrationsseminar" at DHBW Mannheim

Erstellung eines Erklär-Videos mithilfe des packages Manim, in welchem das Thema Neuroevolution anhand einer Replikation der Videospiel-App "The impossible game", bzw. "Geometry Dash" erläutert wird.

Gruppenarbeit (3 Studierende; 2 oder 4 in Ausnahmen)

Mit Manim erstelltes Erklärvideo zu gewähltem Thema (4-5 min pP) als mp4

Theorieteil als PDF (etwa 5 Seiten pP): Grundlagen / Literaturrecherche zum Thema Visualisierung; Gewählten Algorithmus / Verfahren mit Grundlagen detailliert recherchieren / erklären

Manim-Quellcode (hier beilegen oder verlinken) in Repo veröffentlichen: Eigenes Github Repo mit open-source Lizenz oder als PR hier: https://github.com/maschere/manim-algorithmic-vis

## Running the NEAT Algorithm
Tested on Ubuntu inside WSL-2 on Windows 11.
``` python
conda env create --file 0__environment_run_neat.yml  # from src folder
conda activate neat
python 0_continuous_lunar_lander.py
```
- ```0_continuous_lunar_lander.py``` allows the modification of the different probabilities and other parameters of the NEAT algorithm (As well as specifying a different environment and fitness function).
- a run will create a folder inside ```src/runs/``` where the genotypes for each generation and their evaluated fitnesses are saved.
- Using ```1_best_continuous_lunar_lander.py```, different landers can be visualized.
- The complete NEAT algorithm code is in ```src/classes/NEAT.py```.

