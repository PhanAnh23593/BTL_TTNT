"""Backend scaffold. Owner: Bac. Replace with the planned source sections."""
import io
import os

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import transforms

