#! /usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
Copies each reflection level to the clipboard, one at a time.
"""
import pyperclip
import time

levels = [
    "🌟 No thought",
    "🌟🌟 Some unconscious thoughts",
    "🌟🌟🌟 A few conscious thoughts",
    "🌟🌟🌟🌟 Deep reflection",
    "🌟🌟🌟🌟🌟 A bit of effort spent",
    "🌟🌟🌟🌟🌟🌟 Overdid this (eg, overeat, doom scrolling)",
    "🌟🌟🌟🌟🌟🌟🌟 Effort spent, no visible results",
    "🌟🌟🌟🌟🌟🌟🌟🌟 Attaining milestones;",
    "🌟🌟🌟🌟🌟🌟🌟🌟 Holding the fort",
    "🌟🌟🌟🌟🌟🌟🌟🌟🌟 Attaining milestones; holding fort; exploring contradicting perspectives"
]

for level in levels:
    pyperclip.copy(level)
    print(f"Copied: {level}")
    time.sleep(0.4)  # Gives time to stick to clipboard
