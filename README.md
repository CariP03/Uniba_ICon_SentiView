## SentiView
SentiView is a prototype sentiment analysis for English-language IMDb reviews.
The project aims to explore the concept of a Prolog rule-based classifier and compare it with traditional ML models enriched with features extracted from the KB.

## 💾 Setup
Install SWI-Prolog following instructions at the [official site](https://www.swi-prolog.org).

Create a virtual environment in the root folder of the project and activate it:

```
python3 -m venv .venv
source .venv/bin/activate   # on Linux/macOS
.venv\Scripts\activate      # on Windows
```

and install all requirements using the command: 

```pip install -r requirements.txt```

To use Jupyter notebooks, install Jupyter Lab in the virtual environment using:

```pip install jupyterlab```

then move to the folder that contains the notebooks:

```cd notebooks```

and open them using:

```jupyter lab```

## 📀 Download Dataset
The project uses the IMDb reviews dataset, downloadable from [Stanford AI Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). After downloading, extract the archive and place the contents inside a folder named `data` at the root of the project.
