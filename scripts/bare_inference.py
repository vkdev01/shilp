#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import threading
import time
import uuid
from typing import Iterator
import torch
from transformers import (
