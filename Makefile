PY ?= python

.PHONY: demo experiment test
demo:        ## tiny config (2 seeds, 600 backdoor steps); writes to results/demo
	$(PY) -m sleeperlab.experiment --profile demo --out results/demo

experiment:  ## full 5-seed experiment (~15 min on CPU); writes to results/
	$(PY) -m sleeperlab.experiment --profile full

test:
	$(PY) -m pytest -q
