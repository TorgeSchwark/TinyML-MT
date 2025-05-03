import json
from tqdm import tqdm
import uuid
import websocket
import urllib.request
import random

# ----- Model settings -----
BATCH_SIZE = 8
IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 20


# ----- Pipeline settings -----
WORKFLOW = "code_imagegeneration/comfyui/sd_api.json"
CLASSES = ["tomato sauce supermarket packaging", "fruit tea supermarket packaging", "coffee supermarket packaging", "spaghetti supermarket packaging", "coke", "apple", "lemon", "banana", "cucumber", "avocado"]
AMOUNT_PER_CLASS = 160    # Per prompt preset, respects batch size
PATH_PREFIX = "sd/second"

PROMPT_PRESETS = [
    "A high-resolution studio photograph of a single %CLASS% placed centrally on a seamless white background, soft even lighting, minimal shadows",
    "An isolated %CLASS% centered on a plain white backdrop, professional product photography style, sharp focus", # Very good
    "A detailed image of a %CLASS% on a clean white background, centered composition, bright lighting, no reflections",
    "A close-up shot of a %CLASS% against a pure white background, centered, high clarity, minimalistic style",
    "A single %CLASS% displayed on a white background, centered, studio lighting, no shadows, high detail", # Good for coffee
    #"Product image of a supermarket %CLASS% centered on a white background, sharp focus, even lighting, minimal shadows", TOO MANY OBJECTS BUT GOOD ON NON FRUIT
    "A %CLASS% placed in the center of a white background, photographed with soft lighting, no reflections, high-resolution",
    "Studio shot of a supermarket %CLASS% on a white background, centered, clear details, minimal shadows, professional lighting", # Good for coffee
    "An image of a %CLASS% centered on a white background, high-definition, soft shadows, clean appearance",
    #"A centered supermarket %CLASS% on a white background, crisp details, even lighting, no background distractions" NO PACKAGING
]

# This mode allows using a different prompt(s) for each class
PER_PROMPT = True
PER_PROMPT = {
    "avocado": ["A whole avocado uncut, centered on a pure white background, natural lighting, high clarity, professional produce photo for a grocery store"],
    "apple": ["A single fresh red apple placed centrally on a white background, studio lighting, no shadows, supermarket product photo"],
    "lemon": ["A ripe yellow lemon centered on a seamless white background, high-resolution studio photo, minimal shadows, product-style image"],
    "banana": ["A banana bundle placed centrally on a plain white background, professional lighting, clear and detailed, supermarket-style photo"],
    "cucumber": ["A fresh uncut cucumber lying horizontally in the center of a white background, sharp focus, minimal shadows, clean product photography"],
    "tomato sauce": ["A glass jar of tomato sauce with label facing forward, centered on a white background, bright lighting, realistic packaging style photo"],
    "fruit tea": ["A box of fruit tea with visible branding and design, placed in the center on a white background, sharp studio image, supermarket shelf style"],
    "coffee": ["Studio shot of a supermarket coffee on a white background, centered, clear details, minimal shadows, professional lighting"],
    "spaghetti": ["A package of spaghetti with visible label, placed on a white background, product-centered, supermarket-ready photo"],
    "coke": ["A single can of generic coke placed upright in the center of a white background, clean studio lighting, sharp focus, typical product image"]
}

NEGATIVE = ""


# ----- Functions -----
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(ws, prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    prompt_id = json.loads(urllib.request.urlopen(req).read())

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'status':
                queue_remaining = message["data"]["status"]["exec_info"].get("queue_remaining")
                if queue_remaining == 0:
                    break
        else:
            continue

    return prompt_id


# ----- Generate images -----
with open(WORKFLOW, "r") as file:
    prompt = json.load(file)

# Set model settings
prompt["16"]["inputs"]["batch_size"] = BATCH_SIZE
prompt["16"]["inputs"]["width"] = IMAGE_SIZE
prompt["16"]["inputs"]["height"] = IMAGE_SIZE
prompt["15"]["inputs"]["steps"] = NUM_INFERENCE_STEPS
prompt["5"]["inputs"]["text"] = NEGATIVE

ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

amount = AMOUNT_PER_CLASS // BATCH_SIZE

if PER_PROMPT:
    with tqdm(total=len(PER_PROMPT) * amount, desc="Overall Progress") as overall_pbar:
        for cla, prompts in PER_PROMPT.items():
            with tqdm(total=len(prompts) * amount, desc=f"Class: {cla}") as class_pbar:
                for current_prompt in prompts:
                    prompt["8"]["inputs"]["text"] = current_prompt
                    class_name = cla
                    prompt["3"]["inputs"]["filename_prefix"] = f"{PATH_PREFIX}/{class_name}/{class_name}"
                    class_pbar.update(1)
                    overall_pbar.update(1)
                    for i in range(amount):
                        new_seed = random.getrandbits(64)
                        prompt["15"]["inputs"]["seed"] = new_seed
                        prompt["5"]["inputs"]["text"] = NEGATIVE
                        # Queue the prompt for each batch
                        queue_prompt(ws, prompt)
            
            # Queue the prompt for each batch
            for i in range(amount):
                new_seed = random.getrandbits(64)
                prompt["15"]["inputs"]["seed"] = new_seed   
                queue_prompt(ws, prompt)
                overall_pbar.update(1)
else:
    with tqdm(total=len(PROMPT_PRESETS) * len(CLASSES) * amount, desc="Overall Progress") as overall_pbar:
        for cla in CLASSES:
            with tqdm(total=len(PROMPT_PRESETS) * amount, desc=f"Class: {cla}") as class_pbar:
                prompt_nr = 0
                for preset in PROMPT_PRESETS:
                    prompt_nr += 1
                    current_prompt = preset.replace("%CLASS%", cla)
                    prompt["8"]["inputs"]["text"] = current_prompt
                    class_name = cla.replace(" ", "_").replace("supermarket_packaging", "")
                    prompt["3"]["inputs"]["filename_prefix"] = f"{PATH_PREFIX}/{class_name}/{class_name}_{prompt_nr:02d}"
                    # Queue the prompt for each batch
                    for i in range(amount):
                        new_seed = random.getrandbits(64)
                        prompt["15"]["inputs"]["seed"] = new_seed   
                        queue_prompt(ws, prompt)
                        class_pbar.update(1)
                        overall_pbar.update(1)


ws.close()
print("All prompts queued successfully.")