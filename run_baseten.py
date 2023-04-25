# %pip install -U pip
# %pip install accelerate bitsandbytes torch transformers
# %pip install --upgrade baseten

import baseten, truss
baseten.login("r5ueLm1O.hGw6KIlvfhcU5YTEStvEYMfuQmYmkGYB")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers import pipeline

def stablelm_model():
    # Bring in model from huggingface
    return pipeline('text-generation', model='stabilityai/stablelm-tuned-alpha-7b')

model = stablelm_model()

# model_name = "stabilityai/stablelm-tuned-alpha-7b" #@param ["stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-3b", "stabilityai/stablelm-base-alpha-3b"]
# torch_dtype = "float16" #@param ["float16", "bfloat16", "float"]
# load_in_8bit = False #@param {type:"boolean"}
# device_map = "auto"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=getattr(torch, torch_dtype),
#     load_in_8bit=load_in_8bit,
#     device_map=device_map,
#     offload_folder="./offload",
# )

baseten.deploy(
    model,
    model_name='stablelm_model',
)

deployed_model_id = "4q9p535" # See deployed model page to find version ID
model_input = "User: What is the answer to the ultimate question?"

deployed_model = baseten.deployed_model_version_id(deployed_model_id)
response = deployed_model.predict(model_input)
print(response)
# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         stop_ids = [50278, 50279, 50277, 1, 0]
#         for stop_id in stop_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

# system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# - StableLM will refuse to participate in anything that could harm a human.
# """

# prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


# # model_input = "User: What is the answer to the ultimate question?"

# deployed_model = baseten.deployed_model_version_id(deployed_model_id)
# tokens=deployed_model.predict(
#       **inputs,
#   max_new_tokens=64,
#   temperature=0.7,
#   do_sample=True,
#   stopping_criteria=StoppingCriteriaList([StopOnTokens()])
# )
# print(tokenizer.decode(tokens[0], skip_special_tokens=True))