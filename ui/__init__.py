from datetime import datetime
from os import path

import torch
import ipywidgets as widgets
from PIL import Image, ImageDraw, ImageOps, ImageFont


im_per_iter = widgets.Checkbox(
    value=False,
    description='Display image per iteration',
)
display(im_per_iter)

gen_info = widgets.Checkbox(
    value=False,
    description='Include generation info at bottof of image',
)
display(gen_info)

def formatted_now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

save_imgs = widgets.Checkbox(
    value=False,
    description='Save all out images',
)

display(save_imgs)

save_path = widgets.Text(value="", description='Save path:')
display(save_path)

img_name_prefix = widgets.Text(value="formatted_now", description='Name prefix:')
display(img_name_prefix)

prompt = widgets.Textarea(value="", description='Prompt:')
display(prompt)

seed = widgets.IntText(
    value=-1,
    description='Seed:',
    disabled=False
)
display(seed)

steps = widgets.IntSlider(
    value=30,
    step=1,
    description='Steps:',
)
display(steps)

n_iter = widgets.IntSlider(
    value=6,
    min=1,
    max=20,
    step=1,
    description='Total images:',
)
display(n_iter)

batch_size = widgets.IntSlider(
    value=1,
    min=1,
    max=10,
    step=1,
    description='Batch size:',
)
display(batch_size)

cfg_scale = widgets.FloatSlider(
    value=7.5,
    min=1.0,
    max=20.0,
    step=0.25,
    description='Cfg scale:',
    readout_format='.1f',
)
display(cfg_scale)

prior_scale = widgets.IntSlider(
    value=4,
    min=1,
    max=20,
    step=1,
    description='Prior scale:',
)
display(prior_scale)

prior_steps = widgets.IntSlider(
    value=5,
    min=1,
    max=20,
    step=1,
    description='Prior steps:',
)
display(prior_steps)

height = widgets.IntSlider(
    value=768,
    min=128,
    max=4096,
    step=2,
    description='Height:',
)
display(height)

width = widgets.IntSlider(
    value=768,
    min=128,
    max=4096,
    step=2,
    description='Width:',
)
display(width)

sampler = widgets.Dropdown(
    options=['ddim_sampler', 'p_sampler', 'plms_sampler'],
    value='p_sampler',
    description='Sampler:',
    disabled=False,
)
display(sampler)

def add_gen_info(image: Image):

    font = ImageFont.truetype("Roboto-Regular.ttf", size=18)
    text = (
        f"Prompt:   {prompt.value};"
        f"Sampler:   {sampler.value};"
        f"Seed:   {seed.value if seed.value != -1 else torch.seed()};"
        f"Steps:   {steps.value};"
        f"width x height:   {width.value}x{height.value};"
        f"Guidance scale:   {cfg_scale.value};"
        f"prior scale:   {prior_scale.value};"
        f"prior steps:   {prior_steps.value}"
    )
    text_size = font.getsize(text)
    text_width, text_height = text_size
    
    padding = 25
    padded_width = image.width - (padding * 2)
    max_chars_per_line = int(padded_width / font.getsize('x')[0])

    lines = []
    current_line = ""
    for word in text.split(";"):
        if len(current_line + word) + 1 <= max_chars_per_line:
            current_line += word + "     "
        else:
            lines.append(current_line[:-1])
            current_line = word + "     "
    if current_line:
        lines.append(current_line[:-1])

    # Calculate new height
    text_height *= len(lines)
    new_height = image.height + (padding * 2) + text_height + (5 * len(lines))
    
    # Create a new image with the new dimensions
    new_img = Image.new('RGB', (image.width, new_height), color=(255, 255, 255))

    # Paste the original image onto the new image
    new_img.paste(image, (0, 0))

    # Draw text on new image
    y = image.height + padding
    draw = ImageDraw.Draw(new_img)
    for line in lines:
        draw.text((padding, y), line, font=font, fill=(0, 0, 0))
        y += font.getsize(line)[1] + 5
    
    return new_img


def images_processing(images):
    for postfix, img in enumerate(images):
        if gen_info.value:
            img = add_gen_info(img)
        display(img)
        if save_imgs.value:
            prefix = locals().get(img_name_prefix.value, formatted_now)()
            fname = f"{prefix}_{postfix}.png"
            s_path = path.join(save_path.value, fname)
            img.save(s_path)

generate = widgets.Button(description="Generate")
display(generate)