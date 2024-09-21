name: ci

on:
  push:
    branches:
      - master
      - main

permissions:
  contents: write

jobs:
  deploy:
    runs-on: windows-latest  # Change from 'ubuntu-latest' to 'windows-latest'
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.x
      - uses: actions/cache@v2
        with:
          key: ${{ github.ref }}
          path: .cache
      - run: pip install --upgrade --force-reinstall mkdocs-material
      - run: pip install pillow cairosvg
      - run: mkdocs gh-deploy --force
