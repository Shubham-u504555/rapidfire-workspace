PY=python3
VENV?=venv
PORT_SERVE?=5005
MODEL_ID?=runs/sft_ckpt

.PHONY: init install data sft serve curl clean

init:
	test -d $(VENV) || python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip -q install --upgrade pip && $(MAKE) install

install:
	. $(VENV)/bin/activate && pip -q install -r requirements.txt

data:
	. $(VENV)/bin/activate && $(PY) scripts/01_prepare_data.py

sft:
	. $(VENV)/bin/activate && $(PY) scripts/02_sft_run.py

serve:
	. $(VENV)/bin/activate && MODEL_ID=$(MODEL_ID) PORT=$(PORT_SERVE) $(PY) serve_transformers.py

curl:
	curl -sS -X POST http://127.0.0.1:$(PORT_SERVE)/generate \
		-H 'content-type: application/json' \
		-d '{"prompt":"Say hello in 5 words.","max_new_tokens":32}'

clean:
	rm -rf runs __pycache__ .cache .hf_cache
