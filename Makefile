# U8: from a clean checkout to every table and figure.
# τc is never invented. Provide results/cold_start.json from
# scripts/measure_cold_start.py before reproducing.

PYTHON ?= python3
RESULTS ?= results
FIGURES ?= figures
TICKS ?= 80
CONFIG ?= configs/default.yaml
COLD_START ?= $(RESULTS)/cold_start.json

.PHONY: install reproduce overhead figures

install:
	$(PYTHON) -m pip install --require-hashes -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

reproduce:
	@test -f $(COLD_START) || (echo "$(COLD_START) is missing. Measure τc with: $(PYTHON) scripts/measure_cold_start.py" && exit 1)
	$(PYTHON) -m pact.eval.reproduce \
		--config $(CONFIG) \
		--cold-start $(COLD_START) \
		--results $(RESULTS) \
		--figures $(FIGURES) \
		--ticks $(TICKS) \
		$(if $(SEEDS),--seeds $(SEEDS),)

overhead:
	@test -f $(COLD_START) || (echo "$(COLD_START) is missing. Measure τc with: $(PYTHON) scripts/measure_cold_start.py" && exit 1)
	$(PYTHON) -m pact.eval.overhead \
		--config $(CONFIG) \
		--cold-start $(COLD_START) \
		--results $(RESULTS) \
		--ticks $(TICKS)

figures:
	$(PYTHON) make_figures.py --results $(RESULTS) --out $(FIGURES)
