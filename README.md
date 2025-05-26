# Reflective SLM Agent

This project implements a local, self-reflective AI agent powered by a Small Language Model (SLM). It requires no cloud APIs and can be run entirely on your local machine using 'llama.cpp'.

## What It Does

- Loads a quantized small language model using 'llama-cpp-python'
- Answers user questions based on PDF documents indexed in a vector store
- Reflects on the quality of its own answers and suggests improvements
- Logs all interactions to a persistent memory file (`logbook.jsonl`)
- Runs entirely offline

## Requirements

Python 3.9 to 3.13 is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
