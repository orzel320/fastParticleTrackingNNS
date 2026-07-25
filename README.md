# fastParticleTrackingNNS

## O projekcie
To repozytorium jest poświęcone badaniu algorytmów **przybliżonego wyszukiwania najbliższych sąsiadów (ANN)** oraz **uczenia maszynowego (ML)** do rozwiązywania problemu rekonstrukcji torów cząstek w fizyce wysokich energii (High Energy Physics, HEP).

Tradycyjne algorytmy śledzenia (np. kombinatoryczny filtr Kalmana) słabo skalują się przy dużej gęstości cząstek (pile-up). Ten projekt proponuje alternatywny pipeline: sprowadzenie problemu do wyszukiwania na grafach z późniejszym filtrowaniem fałszywych połączeń.

## 🛠 Jak to działa (Pipeline)
Proces podzielony jest na kilka kluczowych etapów:
1. **Generowanie grafu kandydatów (NNS):** Przestrzeń hitów (punktów pozostawionych przez cząstki w detektorze) jest skanowana za pomocą szybkich algorytmów ANN (FAISS, HNSW, cKDTree). Pozwala to w ułamku sekundy połączyć sąsiadujące hity i zbudować pierwotny graf możliwych torów, unikając kwadratowej złożoności pełnego przeglądu ($O(N^2)$).
2. **Ekstrakcja cech (Feature Engineering):** Dla każdej pary połączonych hitów obliczane są cechy geometryczne i fizyczne.
3. **Filtrowanie połączeń (ML Classification):** Zbudowane krawędzie są klasyfikowane jako prawdziwe (należące do jednego toru) lub fałszywe (przypadkowy szum) za pomocą modeli uczenia maszynowego opartych na zespołach (`XGBoost`, `LightGBM`, `Random Forest`).

## 📊 Cele benchmarkingu
W ramach projektu badany jest kompromis pomiędzy szybkością a dokładnością (Recall vs. QPS):
* Porównanie dokładnych algorytmów (Exact kNN) i metod przybliżonych (IVFFlat, IVFPQ, HNSW) na architekturach CPU i GPU.
* Ocena wpływu wymiarowości przestrzeni (w tym po zastosowaniu PCA) na jakość wyszukiwania sąsiadów.


Szybkie wyszukiwanie sąsiadów (k-NN / ANN) i klasyfikacja par hitów w rekonstrukcji torów
cząstek (LHC). Projekt syntetyzuje dane detektora, porównuje dokładne i przybliżone metody
wyszukiwania najbliższych sąsiadów (scipy/sklearn/FAISS/HNSW, CPU i GPU), a następnie uczy
klasyfikator par (Random Forest / XGBoost / LightGBM) i porównuje kompletne potoki
end-to-end pod kątem jakości fizycznej (purity × efficiency) i czasu wykonania.

## Wymagania

- Python **3.9+**
- system Linux/macOS/Windows — cała podstawowa funkcjonalność działa **wyłącznie na CPU**,
  GPU jest opcjonalne (patrz niżej)

## Szybki start

```bash
git clone https://github.com/orzel320/fastParticleTrackingNNS.git
cd fastParticleTrackingNNS

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

make all
```

`make all` samo w sobie: instaluje pakiet (`pip install -e .[dev]`), czyści stare artefakty,
generuje wszystkie zbiory danych, generuje kandydatów par (k-NN) i uruchamia testy
jednostkowe. Po tej jednej komendzie repozytorium jest w pełni gotowe do pracy.

Jeśli wolisz wykonywać kroki osobno:

```bash
make install     # pip install -e .[dev]
make generate    # generuje dane syntetyczne do data/ (Easy/Hard, 1k-1M hitów)
make candidates  # generuje kandydatów par (k-NN) potrzebnych do klasyfikacji
make test        # uruchamia pełny zestaw testów jednostkowych (pytest)
make clean       # usuwa wygenerowane dane i cache
```

## GPU (opcjonalnie, wykrywane automatycznie)

Domyślna instalacja (`pip install -e .`) używa `faiss-cpu` i działa na każdej maszynie.
Klasy `FaissExact`, `FaissIVFFlat` i `FaissIVFPQ` mają parametr `use_gpu` z **auto-detekcją**:

```python
model = FaissIVFFlat(nlist=100, nprobe=10)   # use_gpu=None (domyślnie) = auto-detekcja
```

- `use_gpu=None` (domyślne) — jeśli na maszynie jest dostępne działające GPU, zostanie
  użyte automatycznie; jeśli nie ma GPU, kod po cichu przechodzi na CPU, bez żadnej
  dodatkowej konfiguracji.
- `use_gpu=True` — wymusza GPU; jeśli go nie ma, dostajesz od razu czytelny błąd zamiast
  niejasnego wyjątku z FAISS/CUDA w głębi stosu.
- `use_gpu=False` — wymusza CPU, nawet jeśli GPU jest fizycznie dostępne (przydatne np. do
  uczciwego pomiaru czasu CPU w benchmarkach).

Jeśli masz kartę NVIDIA z zainstalowanym CUDA Toolkit i chcesz też przetestować warianty
GPU (CuPy, FAISS-GPU, crossover CPU/GPU z notatnika `ann_experiment.ipynb`), zainstaluj
dodatkowo:

```bash
pip uninstall faiss-cpu -y
pip install -e ".[gpu]"
```

Zwróć uwagę, że `cupy` wymaga wersji dopasowanej do Twojego CUDA (w `pyproject.toml`
domyślnie ustawiono `cupy-cuda12x` — jeśli masz inną wersję CUDA, podmień na odpowiedni
pakiet, zob. [dokumentację CuPy](https://docs.cupy.dev/en/stable/install.html)).
`faiss-gpu` bywa niestabilne przez pip — w razie problemów zainstaluj przez conda:
`conda install -c pytorch -c nvidia faiss-gpu`.

**Cały pipeline (generowanie danych, testy, klasyfikacja, `final_pipeline.ipynb`) działa
poprawnie bez GPU** — akceleracja GPU jest używana automatycznie, jeśli jest dostępna, ale
nigdy nie jest wymagana.

## Struktura projektu

```
src/hep_tracking/
├── config.py                # konfiguracje (symulacja, dataset, modele)
├── data.py                  # generacja syntetycznych torów i szumu
├── dataset.py                # klasa TrackDataset (X, y, event_ids)
├── generate_candidates.py    # generowanie kandydatów par (k-NN) do klasyfikacji
├── features.py                # cechy par hitów, budowa zbioru par, podział train/val/test
├── models.py                  # implementacje k-NN / ANN (NumPy, CuPy, scipy, sklearn, FAISS, HNSW)
├── classifiers.py             # wrappery RandomForest / XGBoost / LightGBM + strojenie
├── pipeline.py                 # 3 potoki end-to-end: geometryczny / ML / all-pairs
├── plots.py                    # wykresy diagnostyczne i wykresy końcowe (m.in. silver-bullet)
├── benchmark.py                 # narzędzia do pomiaru czasu wykonania
└── utils.py                     # pomocnicze metryki i narzędzia

notebooks/
├── benchmark.ipynb                 # Sprint 1: dokładny k-NN, sanity plots, skalowanie
├── benchmark_pca3d.ipynb           # Eksperyment: redukcja wymiarowości (PCA) a szybkość k-NN
├── ann_experiment.ipynb            # Sprint 2: FAISS/HNSW, recall vs QPS, crossover CPU/GPU
├── classification_experiment.ipynb # Sprint 3: klasyfikacja par (RF/XGBoost/LightGBM)
└── final_pipeline.ipynb            # Sprint 4: porównanie end-to-end, wykres silver-bullet

tests/                        # testy jednostkowe (pytest) dla każdego modułu w src/
```

## Praca z notatnikami

Notatniki w `notebooks/` warto przechodzić w kolejności (1→5) — każdy zakłada, że dane z
`make generate` / `make candidates` już istnieją w `data/`. Uruchom Jupyter standardowo:

```bash
jupyter notebook notebooks/
```

lub w VS Code / JupyterLab — każdy notatnik sam dodaje `src/` do `sys.path`, więc nie
wymaga dodatkowej konfiguracji poza aktywnym środowiskiem wirtualnym.

## Testy

```bash
make test
# lub bezpośrednio:
pytest tests/ -v
```

Wszystkie testy przechodzą na czystym CPU, bez żadnych zewnętrznych zależności poza tymi
zainstalowanymi przez `make install`.
