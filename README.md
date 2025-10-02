# Disaster-Tweet Dataset Builder for Label Studio

A one-command pipeline that:

1. Downloads ~2 M Bluesky (Twitter-like) posts from Hugging-Face 
   1. Dataset: [alpindale/two-million-bluesky-posts](https://huggingface.co/datasets/alpindale/two-million-bluesky-posts)  
2. Keeps only English posts, de-duplicates them  
3. Auto-labels each post with a disaster-relevant category via keyword rules  
4. Down-samples the pool to a pre-defined 1,000-row distribution  
5. Exports two ready-to-use files:

| File | Purpose |
|---|---|
| `label_studio_import.json` | Drag-and-drop import into [Label Studio](https://labelstud.io) (text + pre-label) |
| `pre_labeled_dataset.csv` | Plain CSV backup / inspection | <br>


> Note: This was just a mini-project to simplify our lives for one part of a larger project my senior design capstone and I are working on. <br> <br>
> All this does is generate a potential dataset with a LOW-ACCURACY predicted true_label. <br>
> This is not our main codebase for the project, it just simplifies a smaller step we need to take to benchmark and fine-tune our LLMs. 
<br>

## 12-Category schema

| Label | Description | Target count |
|---|---|---|
| `not_relevant` | General chatter, non-disaster related | 320 |
| `auto_accident` | Crashes, collisions | 110 |
| `fire` | Wildfires, building fires, car-fires, explosions | 100 |
| `flood` | Flooding, flash floods | 100 |
| `severe_storm` | Thunderstorms, hailstorms, wind-storms | 90 |
| `earthquake` | Earthquakes, tremors, aftershocks | 80 |
| `shooting` | Gun violence, active shooter, mass shootings | 70 |
| `tornado` | Tornadoes, twisters | 40 |
| `hurricane` | Hurricanes only | 30 |
| `extreme_heat` | Heat waves, drought, extreme temperatures | 30 |
| `tropical_storm` | Tropical storms, typhoons, monsoons | 20 |
| `other_disaster` | Avalanche, landslide, volcano, tsunami, etc. … | 10 |

<br>

## macOS (Apple-Silicon) quick start

This guide assumes **Python ≥3.9**, **Homebrew**, and **VS Code terminal**.

> Note: I'm running this in VS Code on a MacBook Air, chip: Apple M1, macOS: Ventura 13.1

<br>

### 1. Clone / download this repo and open it in your IDE

```bash
git clone https://github.com/YOUR_USERNAME/cp-1k-dataset-builder.git
cd cp-1k-dataset-builder
```

### 2. Create & activate a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python libraries
You can check requirements.txt that pins every package the pipeline needs (pandas, datasets, Django, openai, etc.). \
From the repo root in the VS Code terminal:
```bash
pip3 install -r requirements.txt
```
That single command installs:
```bash
pandas>=2.2.3
datasets>=2.10.0
Django>=4.2.13
appdirs>=1.4.3
attrs>=19.2.0
bleach>=5.0.0
colorama>=0.4.4
defusedxml>=0.7.1
numpy>=1.26.4
openai>=1.10.0
```
No extra flags are required, pip will fetch the newest compatible wheels.


### 4. Build the 1k dataset
```bash
python3 create_dataset.py
# → label_studio_import.json
# → pre_labeled_dataset.csv
```

### 5. Ensure Homebrew is installed
```bash
which brew
```

### 6. Install Label Studio (full GUI)
Label Studio ships with a hard dependency (psycopg2-binary) that needs PostgreSQL headers on Apple-Silicon.
Fix once:
```bash
# 5a. PostgreSQL client headers
brew install postgresql              # adds pg_config to PATH

# 5b. Force a pre-compiled wheel
# This re-installs the psycopg2-binary from a wheel so no compilation is required
pip3 install --force-reinstall --only-binary :all: psycopg2-binary

# 5c. Now install most recent version of Label Studio
# Because psycopg2-binary is already satisfied, pip will skip it
pip3 install -U label-studio
``` 

Add its binary folder to your shell (permanent):
```bash
echo 'export PATH="$HOME/Library/Python/3.9/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Confirm the binary is there; verify shell now has the updated path:
```bash
which label-studio
```
This should print something like: /Users/username/Library/Python/3.9/bin/label-studio \
Now, every new terminal (inside or outside VS Code) will know where label-studio lives.

### 7. Launch Label Studio
```bash
label-studio
```

Your browser opens http://localhost:8080. \
When creating a new project:
Signin → Create a project → Import label_studio_import.json → start reviewing / correcting the pre-labels.

## Other platforms
- Linux: sudo apt install libpq-dev then continue from step 5b.
- Windows: use the official Label-Studio installer or Docker image; the dataset script also runs in WSL or native Python alike.

## Consider Customizing 
| What to change        | File                                           |
| --------------------- | ---------------------------------------------- |
| Categories / keywords | `keyword_patterns` dict in `create_dataset.py` |
| Desired distribution  | `ideal_distribution_1000` dict                 |
| Language filter       | regex inside `is_english()`                    |


<!-- # pip3 install pandas datasets
# python3 create_dataset.py
# - that'll run the pre-labeled dataset script

# pip3 install label-studio --no-deps
# pip3 install pandas datasets openai "Django>=4.2.13"
# pip3 install -r requirements.txt

# pip3 install label-studio
# - if you get teh psycopg2-binary error, try installing the minimal version first
# pip3 install label-studio-core
# label-studio

# Install without the problematic dependency
#  pip3 install label-studio --no-deps
#  pip3 install django rq django-rq pandas numpy openai\
    
#####

# THIS IS WITHIN VSCODE ON A MacBook Air, chip: Apple M1, macOS: Ventura 13.1
# Ensure Homebre is available inside this terminal
#    which brew
# Install the PostgreSQL client headers 
#    brew install postgresql

# Re-install psycopg2-binary from a wheel so no compilation is required
#    pip3 install --force-reinstall --only-binary :all: psycopg2-binary
# Now try installing label studio again; because psycopg2-binary is already satisfied, pip will skip it
#    pip3 install --upgrade label-studio
# Start Label Studio
#    label-studio
# That should open http://localhost:8080 in your browser

# Add the missing directory to PATH for this session
#    export PATH="$HOME/Library/Python/3.9/bin:$PATH"
# Confirm the binary is there
#    which label-studio
# → /Users/bonszai/Library/Python/3.9/bin/label-studio

# to keep the fix permanent, append the export line to your shell profile
#.   echo 'export PATH="$HOME/Library/Python/3.9/bin:$PATH"' >> ~/.zshrc
#.   source ~/.zshrc

# veryify shell now has the updated PATH
#.   which label-studio
# should print something like: /Users/bonszai/Library/Python/3.9/bin/label-studio
# now can start Label Studio whenever -->
