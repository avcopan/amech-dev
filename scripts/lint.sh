#!/usr/bin/env bash

FILES=$(git ls-files '*.py')
pre-commit run black --files ${FILES[@]}
pre-commit run ruff --files ${FILES[@]}
pre-commit run mypy --files ${FILES[@]}
