# Production web process. Uses the app factory in app_integrated.py.
# Single worker: each worker builds the semantic index and loads models, so
# extra workers duplicate memory/cold-start cost (see render.yaml). Scale via
# --threads on a single worker instead.
web: gunicorn "app_integrated:create_app()" --workers 1 --threads 4 --timeout 120
