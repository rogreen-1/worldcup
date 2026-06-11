.PHONY: pipeline elo features model simulate

pipeline: elo features model simulate

elo:
	python src/build_elo.py

features:
	python src/build_features.py

model:
	python src/fit_model.py

simulate:
	python src/simulate_baseline.py 100000
