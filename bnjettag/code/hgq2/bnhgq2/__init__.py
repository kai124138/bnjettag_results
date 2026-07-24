"""bnhgq2 — config-driven HGQ2 rebuild pipeline for the BitNet jet tagger.

Stages: extract → build → port → verify → ebops → convert → (csynth on mulder) → roc.
Every stage keys its results by the config hash (config.cfg_hash).
"""
