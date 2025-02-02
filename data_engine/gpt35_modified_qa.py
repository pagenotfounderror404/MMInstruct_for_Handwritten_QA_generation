# -*- coding: utf-8 -*-
import re
#import httpx
#from openai import OpenAI
import random
import copy
import json
import time
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
root_path = '.'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

def one_ask(prompt):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    # Generate a response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.85,
            temperature=1.0
        )
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


choice_prompt = '''Giving the description of an image and a question list including five questions, you need to design three multiple choice questions related to the <domain>.
For each sample, the meaning of generated question MUST be similar to the question in the provided question list, and you need to output four choices as candidates.
There should be only one choice that is the answer to the question, and this correct choice should be generated according to the description of the image.
These choices should be indexed by capital letters.
The description of the image and question list for you are as follows:
Description: <caption>.
Question: <original_question_list>.
You MUST output the generated question, choices and answer in the following format:
<Q1> {the generated question 1} </Q1> <C1> {the choices you give} </C1> <A1> {the right choice of the question 1} </A1>
<Q2> {the generated question 2} </Q2> <C2> {the choices you give} </C2> <A2> {the right choice of the question 2} </A2>
<Q3> {the generated question 3} </Q3> <C3> {the choices you give} </C3> <A3> {the right choice of the question 3} </A3>
<Q4> {the generated question 3} </Q4> <C4> {the choices you give} </C4> <A4> {the right choice of the question 3} </A4>
'''

def generate_choice(domain, begin_ix):
    captions_path = f'{domain}_caption_1.jsonl'
    generated_questions_path = f'{domain}_choice.jsonl'
    seed_json = f'{root_path}/all_seed/{domain}.json'

    questions_model = []
    try:
        with open(seed_json, "r", encoding='utf-8') as file:
            json_data = json.load(file)
            questions_model = json_data["select"]["Chinese"]
    except Exception as e:
        logger.info('Failed to read question seeds')
        logger.error(e)

    ix = 0
    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            ix += 1
            if ix < begin_ix:
                continue

            questions_model_list = random.sample(questions_model, min(3, len(questions_model)))
            caption_dict = json.loads(line)
            prompt = choice_prompt

            prior_knowledge = str(caption_dict.get("bing_tag", 'Empty'))
            if not prior_knowledge:
                prior_knowledge = "Empty"

            prompt = prompt.replace("<domain>", domain[6:])
            prompt = prompt.replace('<prior_knowledge>', prior_knowledge)
            prompt = prompt.replace("<caption>", caption_dict['qwen_caption'].replace("\n\n","\n"))
            prompt = prompt.replace('<original_question_list>', str(questions_model_list))

            try:
                out = one_ask(prompt)
                logger.info("[Prompt]\n" + prompt)
                logger.info("[Image Path]: " + caption_dict['image_path'] + "\n[Model Output]: \n" + str(out))

                question_dict = {
                    "image_path": caption_dict['image_path'],
                    "qa_raw": str(out),
                    "model_prompt": prompt
                }
                with open(generated_questions_path, 'a', encoding='utf-8') as outfile:
                    json.dump(question_dict, outfile, ensure_ascii=False)
                    outfile.write('\n')

            except Exception as e:
                logger.info(f"{ix}  [ERROR]")
                logger.info("Error info:" + str(repr(e)))
                caption_dict['err'] = str(repr(e))
                logger.info("Error image path:" + caption_dict['image_path'])
                with open(generated_questions_path, 'a', encoding='utf-8') as outfile:
                    json.dump(caption_dict, outfile, ensure_ascii=False)
                    outfile.write('\n')

    logger.info('****done****')
    logger.info(f"Total generated {ix} pairs.")



lqa_prompt = 'Provide a description of an image and a list of multiple questions, you need to design three long question answering questions related to the <domain>.\n\
For each sample, the meaning of generated question MUST be similar to the question in the provided question list, and you need to output a detailed answer to the question.\n\
The detailed answer to this question should be generated based on the description of the image.\n\
The description of the image and question list for you are as follows:\n\
Description: <caption>. \n Question: <original_question_list>. \n  \
You MUST output the generated questions and answers in the following format:\n\
<Q1> {the generated question 1} </Q1> <A1> {the long answer of the question 1} </A1>\n\
<Q2> {the generated question 2} </Q2> <A2> {the long answer of the question 2} </A2>\n\
<Q3> {the generated question 3} </Q3> <A3> {the long answer of the question 3} </A3>\n'


def generate_long_qa(domain, begin_ix=0):
    print("\n\n****start lqa and answer working****\n\n")
    captions_path = f'{domain}_caption_1.jsonl'
    generated_queations_path = f'{domain}_lqa.jsonl'
    error_dest_path = f'{domain}_lqa_err.jsonl'
    seed_json = f'{domain}_question_data.json'

    questions_model = []
    with open(seed_json, "r", encoding='utf-8') as file:
        try:
            json_data = json.load(file)
            questions_model = json_data["long_qa"]["English"]
        except:
            print('not found')

    ix = 0
    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            ix += 1
            if ix < begin_ix:
                continue
            
            questions_model_list = random.sample(questions_model, min(3, len(questions_model)))
            print("questions_model_list: " + str(questions_model_list))

            caption_dict = json.loads(line)
            prompt = lqa_prompt
            prompt = prompt.replace("<domain>", domain)
            prompt = prompt.replace('<prior_knowledge>', str(caption_dict.get("prior", 'Empty')))
            prompt = prompt.replace("<caption>", caption_dict['qwen_caption'])
            prompt = prompt.replace('<question_templates>', str(questions_model_list))
            try:
                out = one_ask(prompt)
                print("[image_path]: \n" + caption_dict['image_path'] + "\n\n[GPT OUT]: \n" + str(out))
                question_dict = {
                    "image_path": caption_dict['image_path'], 
                    "qa_raw": str(out),
                    "gpt_prompt": prompt    
                }
                open(generated_queations_path, 'a', encoding='utf-8').write(
                        json.dumps(question_dict, ensure_ascii=False)+'\n'
                    )
            except Exception as e:
                print(str(ix) + "  [ERROR]")
                print("error info:" + str(repr(e)))
                caption_dict['err'] = str(repr(e))
                print("error image path:" + caption_dict['image_path'])
                open(error_dest_path, 'a', encoding='utf-8').write(
                    json.dumps(caption_dict, ensure_ascii=False)+'\n'
                )

    print('****done****')
    print("total generate " + str(ix) + " pairs. ")


sqa_prompt = 'Provide a description of an image and a list of multiple questions, you need to desigin three short question answering questions related to the <domain>.\n\
For each sample, the meaning of generated question MUST be similar to the question in the provided question list, and you need to output a few words or short sentences as a short answer to the question.\n\
The answer to this question should be generated based on the description of the image.\n\
The description of the image and question list for you are as follows:\n\
Description: <caption>. \n Question: <original_question_list>. \n  \
You MUST output the generated questions and answers in the following format:\n\
<Q1> {the generated question 1} </Q1> <A1> {the short answer of the question 1} </A1>\n\
<Q2> {the generated question 2} </Q2> <A2> {the short answer of the question 2} </A2>\n\
<Q3> {the generated question 3} </Q3> <A3> {the short answer of the question 3} </A3>\n'



def generate_short_qa(domain, begin_ix=0):
    print("\n\n****start sqa and answer working****\n\n")
    captions_path = f'{domain}_caption_1.jsonl'
    generated_queations_path = f'{domain}_sqa.jsonl'
    error_dest_path = f'{domain}_sqa_err.jsonl'
    seed_json = f'{domain}_question_data.json'

    questions_model = []
    with open(seed_json, "r", encoding='utf-8') as file:
        try:
            json_data = json.load(file)
            questions_model = json_data["short_qa"]["English"]
        except:
            print('seed not found')

    ix = 0
    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            ix += 1
            if ix < begin_ix:
                continue

            questions_model_list = random.sample(questions_model, min(3, len(questions_model)))
            caption_dict = json.loads(line)

            prompt = sqa_prompt
            prompt = prompt.replace("<domain>", domain)
            prompt = prompt.replace('<prior_knowledge>', str(caption_dict.get("prior", 'Empty')))
            prompt = prompt.replace("<caption>", caption_dict['qwen_caption'])
            prompt = prompt.replace('<question_templates>', str(questions_model_list))
            try:
                out = one_ask(prompt)
                print("[image_path]: \n" + caption_dict['image_path'] + "\n\n[GPT OUT]: \n" + str(out))
                question_dict = {
                    "image_path": caption_dict['image_path'], 
                    "qa_raw": str(out),
                    "gpt_prompt": prompt    
                }
                open(generated_queations_path, 'a', encoding='utf-8').write(
                    json.dumps(question_dict, ensure_ascii=False)+'\n'
                )
            except Exception as e:
                print(str(ix) + "  [ERROR]")
                print("error info:" + str(repr(e)))
                caption_dict['err'] = str(repr(e))
                print("error image path:" + caption_dict['image_path'])
                open(error_dest_path, 'a', encoding='utf-8').write(
                    json.dumps(caption_dict, ensure_ascii=False)+'\n'
                )

    print('****done****')
    print("total generate " + str(ix) + " pairs. ")



judge_prompt = 'Provide a description of an image and a list of multiple questions, you need to desigin three true or false questions related to the <domain>.\n\
For each sample, the meaning of generated question MUST be similar to the question in the provided question list, and you need to output "Yes" or "No" as the answer to the question.\n\
The answer to this question should be generated based on the description of the image.\n\
The description of the image and question list for you are as follows:\n\
Description: <caption>. \n Question: <original_question_list>. \n  \
You MUST output the generated questions and answers in the following format:\n\
<Q1> {the generated question 1} </Q1> <C1> {"Yes",  "No"} </C1> <A1> {the right choice of the question 1} </A1>\n\
<Q2> {the generated question 2} </Q2> <C2> {"Yes",  "No"} </C2> <A2> {the right choice of the question 2} </A2>\n\
<Q3> {the generated question 3} </Q3> <C3> {"Yes",  "No"} </C3> <A3> {the right choice of the question 3} </A3>\n'


def generate_judge(domain, begin_ix=0):
    print("\n\n****start judge and answer working****\n\n")
    captions_path = f'{domain}_caption_1.jsonl'
    generated_queations_path = f'{domain}_judge.jsonl'
    seed_json = f'{domain}_question_data.json'

    questions_model = []
    with open(seed_json, "r", encoding='utf-8') as file:
        try:
            json_data = json.load(file)
            questions_model = json_data["judge"]["English"]
        except:
            logger.info('读取问题种子失败')
            return

    ix = 0
    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            ix += 1
            if ix < begin_ix:
                continue

            questions_model_list = random.sample(questions_model, min(3, len(questions_model)))
            caption_dict = json.loads(line)

            prompt = judge_prompt
            prompt = prompt.replace("<domain>", domain[6:])
            prior_knowledge = str(caption_dict.get("bing_tag", 'Empty'))
            if prior_knowledge == "":
                prior_knowledge = "Empty"
            prompt = prompt.replace('<prior_knowledge>', prior_knowledge)
            prompt = prompt.replace("<caption>", caption_dict['qwen_caption'].replace("\n\n","\n"))
            prompt = prompt.replace('<question_templates>', str(questions_model_list))
            try:
                out = one_ask(prompt)
                question_dict = {
                    "image_path": caption_dict['image_path'], 
                    "qa_raw": str(out),
                    "gpt_prompt": prompt    
                }
                open(generated_queations_path, 'a', encoding='utf-8').write(
                    json.dumps(question_dict, ensure_ascii=False)+'\n'
                )
                
            except Exception as e:
                logger.info(str(ix) + "  [ERROR]")
                logger.info("error info:" + str(repr(e)))
                caption_dict['err'] = str(repr(e))
                logger.info("error image path:" + caption_dict['image_path'])
                open(generated_queations_path, 'a', encoding='utf-8').write(
                        json.dumps(caption_dict, ensure_ascii=False)+'\n'
                    )

    logger.info('****done****')
    logger.info("total generate " + str(ix) + " pairs. ")

if __name__ == "__main__":
    domain = "poster"
    #generate_choice(domain,begin_ix=0)
    #generate_long_qa(domain,begin_ix=0)
    generate_short_qa(domain,begin_ix=0)
