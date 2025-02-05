import json
from PIL import Image
import imghdr
import base64
import io
import logging
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import traceback


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Qwen2-VL-2B-Instruct model and tokenizer
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8", torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8")


def list_to_str(tmp):
    res = ''
    for item in tmp:
        res += '\n' + str(item)
    return res


def load_ocr_results(source_path):
    with open(source_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def one_ask(text, image_path):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        # Preparation for inference
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        logger.info(text)
        logger.info(output_text)

        return output_text
    except Exception as e:
        logger.error(f'[error in one ask]: {repr(e)}')
        time.sleep(1.5)
    logger.error("Failed to generate response.")
    return "error"

caption_prompt = "Please describe the image for me in as much detail as possible. You need to generate a description of at least 120 words. If you can, identify what objects are present in the image."
caption_prompt_text = "This is an image accompanied by text information, with the content of the text being: <ocr_text>. Based on both the image itself and the text content, understand the image and then describe it as comprehensively as possible, generating a description of at least 200 words."

def get_qwen_caption(img_folder, source_path, dest_path, begin_ix):
    ocr_results = load_ocr_results(source_path)
    
    for ix, (image_file, ocr_data) in enumerate(ocr_results.items()):
        if ix < begin_ix:
            continue

        logger.info(f"Processing {ix + 1} of {len(ocr_results)}")
        try:
            text = ocr_data[0]['text'] if ocr_data else ''
            prompt = caption_prompt_text.replace("<ocr_text>", text) if text else caption_prompt

            image_path = os.path.join(img_folder, image_file)
            logger.info(f"Image path: {image_path}")
            qwen_output = one_ask(prompt, image_path)
        
            new_item = {
                'image_path': image_file,
                'ocr_text': text,
                'qwen_caption': qwen_output,
                'qwen_prompt': prompt
            }

            with open(dest_path, 'a', encoding='utf-8') as f:
                json.dump(new_item, f, ensure_ascii=False)
                f.write('\n')
            time.sleep(1.5)
        except Exception as e:
            logger.error(f"[error]: {repr(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    img_folder = '../anno'
    source_path = 'ocr_results.json'
    dest_path = 'anno_caption.jsonl'
    begin_ix = 0
    get_qwen_caption(img_folder, source_path, dest_path, begin_ix)
    logger.info("done.")
