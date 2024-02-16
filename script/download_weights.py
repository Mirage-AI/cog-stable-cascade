#!/usr/bin/env python
import sys

from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

sys.path.append('.')
from predict import DECODER_CACHE, PRIOR_CACHE

StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", cache_dir=PRIOR_CACHE)
StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  cache_dir=DECODER_CACHE)
