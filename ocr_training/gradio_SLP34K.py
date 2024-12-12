import argparse
import glob
import random 
import gradio as gr

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


def recognize_text(image):
    raw_image = image.convert('RGB')
    image_mae = img_transform(raw_image).unsqueeze(0).to(args.device)

    p = model(image_mae).softmax(-1)
    new_pred, p = model.tokenizer.decode(p)
    new_pred = new_pred[0]
    return new_pred

         
if __name__ == '__main__':

    examples = glob.glob('../SLP34K/test/*.jpg')
    random.shuffle(examples)
    examples_img_len = 50 
    random_examples = examples[:examples_img_len]
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    kwargs.update({'mae_pretrained_path': "pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth"})
    print(f'Additional keyword arguments: {kwargs}')
    args.device = "cuda"
    args.checkpoint = "checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt"
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device) # 新模型

    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    iface = gr.Interface(fn=recognize_text,
                     inputs=gr.Image(type='pil', label="upload image"),
                     outputs=gr.Textbox(label="recognized text"),
                     examples=random_examples,
                     title="SLP34K Baseline Text Recognition Example",
                     description="Upload images and view recognized text! Note that this example only supports recognizing English characters.")

    iface.launch()


