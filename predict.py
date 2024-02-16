# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from cog import BasePredictor, Input, Path
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from typing import List

DECODER_CACHE = "./sc-decoder"
PRIOR_CACHE = "./sc-prior"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        device = "cuda"
        self.prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to(device)
        self.decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.float16).to(device)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        width: int = Input(
          description="Width of output image",
          default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        prior_steps: int = Input(
            description="Number of denoising steps for latent generator (prior)", ge=1, le=500, default=20
        ),
        decoder_steps: int = Input(
            description="Number of denoising steps for decoder", ge=1, le=500, default=10
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=4.0
        ),
        agree_to_research_only: bool = Input(
            description="You must agree to use this model only for research. It is not for commercial use.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if not agree_to_research_only:
            raise Exception(
                "You must agree to use this model for research-only, you cannot use this model comercially."
            )
        
        prior_output = self.prior(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_outputs,
            num_inference_steps=prior_steps
        )
        decoder_output = self.decoder(
            image_embeddings=prior_output.image_embeddings.half(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=decoder_steps
        ).images

        output_paths = []
        for i, image in enumerate(decoder_output):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
