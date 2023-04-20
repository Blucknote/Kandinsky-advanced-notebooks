import main
import ui

out = ui.widgets.Output()

@out.capture(clear_output=True)
def txt2img(event):
    if ui.seed.value != -1:
        main.torch.manual_seed(ui.seed.value)
        main.torch.cuda.manual_seed_all(ui.seed.value)
    
    images = []
    for cur_iter in range(ui.n_iter.value):
        image_iter = main.model.generate_text2img(
            ui.prompt.value,
            num_steps=ui.steps.value,
            batch_size=ui.batch_size.value,
            guidance_scale=ui.cfg_scale.value,
            h=ui.height.value,
            w=ui.width.value,
            sampler=ui.sampler.value, 
            prior_cf_scale=ui.prior_scale.value,
            prior_steps=str(ui.prior_steps.value),
            negative_prior_prompt=ui.neg_prior.value,
            negative_decoder_prompt=ui.neg_prompt.value,
        )
        main.torch_gc()
        if ui.im_per_iter.value:
            ui.images_processing(image_iter)
        else:
            images.extend(image_iter)


    if not ui.im_per_iter.value and images:
        ui.images_processing(images)
        
        
ui.generate.on_click(txt2img)
display(out)