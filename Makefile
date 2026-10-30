.PHONY: setup dev test sample

setup:
	./scripts/setup.sh

dev:
	./scripts/dev.sh

test:
	cd backend && .venv/bin/python -m pytest -q

sample:
	backend/.venv/bin/python scripts/make_sample_clip.py storage/sample_clip.mp4
