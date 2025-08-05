import torch
import torch.nn.functional as F
from pipeline_flux_probing import FluxPipeline
from transformer_flux_probing import FluxTransformer2DModel
import string
from diffusers.utils.torch_utils import randn_tensor
from probing_attn_utils import register_probing_control
from PIL import Image
import os
import string
from diffusers.models.attention_processor import FluxAttnProcessor2_0

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
flux_transfomer = FluxTransformer2DModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe.transformer = flux_transfomer

device = "cuda:0"
pipe = pipe.to(device)

def register_ori_attention(model):
    attn_procs = FluxAttnProcessor2_0()
    model.transformer.set_attn_processor(attn_procs)
    print(f"Model {model.transformer.__class__.__name__} is registered attention processor: FluxAttnProcessor2_0")

def resize_and_concat_images(image_list, resize_width=512, resize_height=512):
    resized_images = [
        img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        for img in image_list
    ]
    total_width = len(resized_images) * resize_width
    concat_image = Image.new("RGB", (total_width, resize_height))
    
    for i, img in enumerate(resized_images):
        concat_image.paste(img, (i * resize_width, 0))
    return concat_image

def prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None].to(device)
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :].to(device)

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids

def get_rotary_emb(pipe, height=64, width=64, height_shift=0, width_shift=0, device=device):
    ori_latent_ids = prepare_latent_image_ids(height, width, device, torch.bfloat16)
    shifted_latent_ids = ori_latent_ids + torch.tensor([0, height_shift, width_shift], dtype=torch.bfloat16).to(device)
    text_ids = torch.zeros(512, 3, dtype=torch.bfloat16, device=device)
    final_ids = torch.cat((text_ids, shifted_latent_ids), dim=0)
    rotary_emb = pipe.transformer.pos_embed(final_ids)
    return rotary_emb


prompt_list = [
    "A serene beach with palm trees and a colorful sunset",
    "A futuristic city with flying cars and neon lights",
    "A magical forest with glowing mushrooms and fairies",
    "A snowy mountain with a small cabin and smoke rising from the chimney",
    "A dog on the road",
    "A peaceful garden with a koi pond and blooming flowers",
    "A medieval castle surrounded by a moat and lush greenery",
    "A cozy living room with a fireplace and a fluffy cat on the sofa",
    "A vibrant underwater scene with coral reefs and exotic fish",
    "A quiet library with tall bookshelves and soft lighting",
    "A space station orbiting Earth with astronauts floating",
    "A colorful hot air balloon festival in a green valley",
    "A flock of birds flying across a clear sky",
    "A picturesque village in the mountains with cobblestone streets",
    "A serene desert landscape with sand dunes and a camel",
    "A lively carnival with Ferris wheels and cotton candy stands",
    "A futuristic robot assisting in a kitchen",
    "A golden retriever playing in a park",
    "A tranquil lake with a wooden dock and canoes",
    "A gothic cathedral with intricate stained glass windows",
    "A vibrant street art mural in an urban neighborhood",
    "A mystical dragon soaring over a mountain range",
    "A traditional Japanese tea house surrounded by cherry blossoms",
    "A retro diner with neon signs and a jukebox",
    "A herd of elephants walking through the savanna",
    "A tropical island with turquoise water and white sandy beaches",
    "A high-tech laboratory with futuristic gadgets and holograms",
    "A charming farm with a red barn and grazing animals",
    "A serene waterfall in a dense jungle",
    "A cozy bookstore with shelves of old and new books",
    "A space-themed amusement park with rocket rides and alien mascots",
    "A winter wonderland with ice sculptures and snow-covered trees",
    "A rustic vineyard with grapevines and a small winery",
    "A cat sitting on a windowsill watching the rain",
    "A colorful butterfly garden with flowers of every hue",
    "A Viking village with wooden huts and longboats by the shore",
    "A surreal dreamscape with floating islands and impossible geometry",
    "A panda eating bamboo in a dense forest",
    "A peaceful campsite by a river under a starry sky",
    "A majestic eagle soaring over a canyon at sunrise",
    "A vibrant cityscape during a rainy evening with reflections",
    "A magical library with books flying and glowing scrolls",
    "A steampunk airship cruising above a bustling city",
    "A group of penguins huddling together on ice",
    "A mysterious forest with dense fog and shadowy figures",
    "A serene meadow with wildflowers and a grazing deer",
    "A futuristic subway station with sleek design and neon lighting",
    "A festive Christmas market with twinkling lights and snow",
    "A traditional windmill on a hillside surrounded by tulips",
    "A mystical portal opening in a quiet forest clearing"
]



seed_list = range(5)
height = 1024
width = 1024
shape = (1, 16, height//8, width//8)
for prompt in prompt_list:
    for seed in seed_list:

        output_sample_dir = os.path.join('rope_change_dataset_dir', prompt.translate(str.maketrans('', '', string.punctuation)).replace(' ', '_'), str(seed))
        os.makedirs(output_sample_dir, exist_ok=True)

        if os.path.exists(os.path.join(output_sample_dir, 'sample.jpg')):
            print(os.path.join(output_sample_dir, 'sample.jpg'), 'exist!')
        else:
            generator = torch.Generator("cuda").manual_seed(seed)
            latents = randn_tensor(shape, generator=generator, device=pipe._execution_device, dtype=torch.bfloat16)
            latents = pipe._pack_latents(latents, 1, 16, height//8, width//8)

            register_ori_attention(pipe)
            image_sample_list = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                latents=latents,
            ).images

            resize_and_concat_images(image_sample_list).save(f'{output_sample_dir}/sample.jpg')

        given_v_rotary_emb_list = [None, [10,10], [0,20], [64,0]]
        for i, each_shift_param in enumerate(given_v_rotary_emb_list):
            if i == 0:
                given_v_rotary_emb = each_shift_param
            else:
                given_v_rotary_emb = get_rotary_emb(pipe, height//16, width//16, height_shift=each_shift_param[0], width_shift=each_shift_param[1], device=device)
            output_dir = os.path.join(output_sample_dir, str(each_shift_param).translate(str.maketrans('', '', string.punctuation)).replace(' ', '_'))
            os.makedirs(output_dir, exist_ok=True)
            for idx in range(57):
                if os.path.exists(os.path.join(output_dir, f'{str(idx)}.jpg')):
                    print(os.path.join(output_dir, f'{str(idx)}.jpg'), 'exist!')
                    continue

                generator = torch.Generator("cuda").manual_seed(seed)
                latents = randn_tensor(shape, generator=generator, device=pipe._execution_device, dtype=torch.bfloat16)
                latents = pipe._pack_latents(latents, 1, 16, height//8, width//8)

                processor_args = {
                    "start_step": 0,
                    "start_layer": 0,
                    "layer_idx": [idx],
                    "step_idx": None,
                    "total_layers": 57,
                    "total_steps": 50,
                }

                register_probing_control(pipe, **processor_args)
                image_ori_list = pipe(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    latents=latents,
                    given_v_rotary_emb=given_v_rotary_emb,
                ).images
                resize_and_concat_images(image_ori_list).save(f'{output_dir}/{str(idx)}.jpg')