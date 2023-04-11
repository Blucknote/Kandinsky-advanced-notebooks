from datetime import datetime
from os import path
import ipywidgets as widgets


im_per_iter = widgets.Checkbox(
    value=False,
    description='Display image per iteration',
)
display(im_per_iter)

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

def images_processing(images):
    for postfix, img in enumerate(images):
        display(img)
        if save_imgs.value:
            prefix = locals().get(img_name_prefix.value, formatted_now)()
            fname = f"{prefix}_{postfix}.png"
            s_path = path.join(save_path.value, fname)
            print(s_path)
            img.save(s_path)

generate = widgets.Button(description="Generate")
display(generate)