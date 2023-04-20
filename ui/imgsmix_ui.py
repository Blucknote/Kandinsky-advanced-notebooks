from io import BytesIO

from PIL import Image
import ui
import main

img1 = ui.widgets.FileUpload(
    accept='image/*',
    multiple=False,
)
img2 = ui.widgets.FileUpload(
    accept='image/*',
    multiple=False,
)


preview = ui.widgets.Output()

@preview.capture()
def gen_preview(change):
    pre = ui.widgets.Image(
        value=change["new"][0]["content"],
        format='png',
        width=350,
        height=350,
    )
    display(pre)

img1.observe(gen_preview, names='value')
img2.observe(gen_preview, names='value')

display(preview)

# display(ui.widgets.Box([preview1, preview2]))

def get_pills():
    raw1 = BytesIO(img1.value[0]["content"].tobytes())
    pil1 = Image.open(raw1)
    raw2 = BytesIO(img2.value[0]["content"].tobytes())
    pil2 = Image.open(raw2)
    return [pil1, pil2]

img_w1 = ui.widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=2.0,
    step=0.1,
    description='IMG1 weigth:',
    readout_format='.1f',
)


img_w2 = ui.widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=2.0,
    step=0.1,
    description='IMG2 weigth:',
    readout_format='.1f',
)

img_box1 = ui.widgets.widgets.VBox([img1, img_w1])
img_box2 = ui.widgets.widgets.VBox([img2, img_w2])

imgs_box = ui.widgets.HBox([img_box1, img_box2])
display(imgs_box)

def center_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

out = ui.widgets.Output()

@out.capture(clear_output=True)
def imgmix(event):
    if ui.seed.value != -1:
        main.torch.manual_seed(ui.seed.value)
        main.torch.cuda.manual_seed_all(ui.seed.value)
    
    images = []
    for cur_iter in range(ui.n_iter.value):
        image_iter = main.model.mix_images(
            get_pills(),
            [img_w1.value, img_w2.value], 
            num_steps=ui.steps.value,
            batch_size=ui.batch_size.value,
            guidance_scale=ui.cfg_scale.value,
            h=ui.height.value,
            w=ui.width.value,
            sampler=ui.sampler.value, 
            prior_cf_scale=ui.prior_scale.value,
            prior_steps=str(ui.prior_steps.value),
        )
        main.torch_gc()
        if ui.im_per_iter.value:
            ui.images_processing(image_iter)
        else:
            images.extend(image_iter)


    if not ui.im_per_iter.value and images:
        ui.images_processing(images)

ui.prompts.close()
ui.generate.close()

mix = ui.widgets.Button(description="Mix")
mix.on_click(imgmix)
display(mix)
display(out)
