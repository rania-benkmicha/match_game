# Match_game
A project that simulates football matches using generative AI, leveraging exploratory data analysis and three distinct approaches to handle various game scenarios. Designed to explore strategic outcomes and dynamic gameplay modeling.

## Prerequisites
- [Conda] ( environment management)
- [Git]( cloning the repository)

## Installation

### Clone the repository:
```bash
git clone 
cd match_test/Use_Case/


```
### Create a Conda environment :
```bash
conda create --name name_of_env python=3.9.12
```
### Activate the Conda environment :

```bash
conda activate name_of_env
```

### Install the required packages using pip :
```bash
pip install -r requirement.txt
```
### running project :
```bash
python test_main.py --file_name 'match_1.json' --path './data_match' --seed ['run','shot','shot','cross'] --set_action ['walk','run','dribble','rest','pass','tackle','shot','cross'] --match_length 5 --number_of_games 1 --playstyle_actions ["shot", "sprint","pass"]
```
