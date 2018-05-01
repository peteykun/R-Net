# Exploring Defenses for Reading Comprehension Systems

Course project by Soham Pal and Akash Valsangkar for the course **E1 246: Natural Language Understanding** offered at IISc Bangalore. The code is based on the repository [R-Net](https://github.com/HKUST-KnowComp/R-Net).

Note that the original network must be trained before running the notebooks via `python config.py --mode prepro` and `python config.py --mode train`. You must create the directories `log/binary_model`, `log/badptr_model` and `log_combo_model` after this. Then, run the data processing notebooks. Note that the notebook `4 Data Processing (Combined).ipynb` must be run before running any of the other `Data Processing` notebooks. Finally, you can run the experiments by first running the `Training` notebook and then the corresponding `Tester` notebook.
