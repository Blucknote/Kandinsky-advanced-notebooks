from io import BytesIO

from PIL import Image
import ui
import main


style = {'description_width': 'initial', 'padding': '10px'}
mix = ui.widgets.Button(description="Mix")

constructor = ui.widgets.Output()

class ImageUploadConstructor:
    upload_buttons = []
    weights = []
    boxes = {}
    
    @constructor.capture()
    def create_button(self, event):
        btn = ui.widgets.FileUpload(
            accept='image/*',
            multiple=False,
        )
        self.upload_buttons.append(btn)
        weight = ui.widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=2.0,
            step=0.1,
            description='IMG weigth:',
            readout_format='.1f',
        )
        self.weights.append(weight)

        btn.observe(self.generate_preview, names='value')
        
        deleter = ui.widgets.Button(description="Delete image from mix", style=style)
        deleter.on_click(self.delete)
        
        box = ui.widgets.widgets.VBox([ui.widgets.Output(), btn, weight, deleter])
        display(box)
        self.boxes[id(btn)] = box
        mix.description = f"Mix ({len(self.upload_buttons)}) images"
        
    def generate_preview(self, change):
        out = self.boxes[id(change["owner"])].children[0]
        out.clear_output(wait=True)
        pre = ui.widgets.Image(
            value=change["new"][0]["content"],
            format='png',
            width=350,
            height=350,
        )
        with out:
            display(pre)
    
    def get_pills(self):
        pills = []
        for notpill in self.upload_buttons:
            _bytes = BytesIO(notpill.value[0]["content"].tobytes())
            pills.append(Image.open(_bytes))
        return pills
    
    def get_weights(self):
        return [w.value for w in self.weights]
    
    def delete(self, event):
        for btn, box in self.boxes.items():
            if box.children[-1] is event:
                box.close()
                break
                
        for index, upload_btn in enumerate(self.upload_buttons):
            if id(upload_btn) == btn:
                break
                
        self.upload_buttons.pop(index)
        self.weights.pop(index)
        mix.description = f"Mix ({len(self.upload_buttons)}) images"


display(constructor)
uploader_constructor = ImageUploadConstructor()
add_img = ui.widgets.Button(description="Add image to mix")
uploader_constructor.create_button(None)
uploader_constructor.create_button(None)
add_img.on_click(uploader_constructor.create_button)
display(add_img)
    
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
            uploader_constructor.get_pills(),
            uploader_constructor.get_weights(), 
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

mix.on_click(imgmix)
display(mix)
display(out)
