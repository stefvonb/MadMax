# -*- coding: utf-8 -*- 

VERSION = 0.1

FP_TOLERANCE = 1E-05

import os
MODULE_LOC = os.path.realpath(__file__).replace("/madmax/__init__.py", "")

import subprocess

try:
    git_description = subprocess.check_output(["git", "--git-dir={}/.git".format(MODULE_LOC), "describe", "--always"]).strip().decode("utf-8")
except subprocess.CalledProcessError:
    git_description = "Unknown"

print("Using the new maximisation MEM method (codename MadMax)\nGit description: {}".format(git_description))

logo = """
    ▞▍    ▞▍     ▞▍ ▛▀▀▚     ▞▍    ▞▍     ▞▍ ▚   ▞
   ▞ ▍   ▞ ▍    ▞ ▍ ▍   ▍   ▞ ▍   ▞ ▍    ▞ ▍  ▚ ▞
  ▞  ▍  ▞  ▍   ▟▃▃▍ ▍   ▍  ▞  ▍  ▞  ▍   ▟▃▃▍   █
 ▞   ▍ ▞   ▍  ▞   ▍ ▍   ▍ ▞   ▍ ▞   ▍  ▞   ▍  ▞ ▚
▞    ▙▞    ▍ ▞    ▍ ▙▄▄▞ ▞    ▙▞    ▍ ▞    ▍ ▞   ▚
"""
import sys
sys.stdout.write("\033[1;31m")
print(logo)
sys.stdout.write("\033[0;0m")

print("As for you, go forth and maximise...")

class PhysicsError(Exception):
    pass

import numpy as np
np.seterr(invalid="ignore")

try:
    import lhapdf
    lhapdf.setVerbosity(0)
except ImportError:
    print("WARNING:\tLHAPDF not found. Check your installation...")

