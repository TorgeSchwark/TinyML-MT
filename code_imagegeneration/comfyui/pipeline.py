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
inactive = ["tomato sauce supermarket packaging", "fruit tea supermarket packaging", "coffee supermarket packaging", "spaghetti supermarket packaging", "coke", "apple", "lemon", "banana", "cucumber", "avocado"]
CLASSES = ["background3"]
AMOUNT_PER_CLASS = 8    # Per prompt preset, respects batch size
PATH_PREFIX = "sd/directions"
PER_PROMPT_MODE = True  # This mode allows using a different prompt(s) for each class

# PROMPTS (chaotic :D)
# Background prompts needs to be filtered again
background = [ #! FILTER
    # 1. Minimal, clean backgrounds
    "A top-down view of a clean white background, soft shadows, minimal texture, ideal for product photography, high-resolution",
    "Flat beige surface with subtle lighting, studio photography background, no objects, minimal texture",
    "Top-down view of a light gray matte background, smooth and clean, professional product setup",
    "Minimal wood grain tabletop, natural lighting from above, empty and clean, light birch color",

    # 2. Light structured surfaces
    "Textured paper surface, soft cream color, flat lay perspective, clean and minimal",
    "Concrete background with smooth finish, subtle lighting, top-down view, industrial tone",
    "Laminated kitchen countertop, white with light speckles, flat lay view, evenly lit",

    # 3. Basket/cart-like contexts
    "Interior of a red plastic shopping basket, top-down view, empty, realistic lighting",
    "Top view of a metal shopping cart base, silver wireframe pattern, clean and empty",
    "Dark plastic shopping basket interior, flat lay, realistic textures, good lighting for placing items",

    # 4. Checkout conveyor belt and context
    "Black rubber checkout conveyor belt with subtle grooves, top-down view, empty, soft shadows",
    "Supermarket checkout counter surface, empty, barcode scanner and divider visible, top-down photography",
    "Flat lay of a cashier conveyor with no products, rubber surface and checkout divider, realistic setting",

    # 5. Store floor / ambient realism
    "Blurred supermarket shelf background, bokeh effect, top-down floor surface in foreground, realistic",
    "Grocery store aisle with blurred background, light gray floor in sharp focus, top-down view",
    "Empty supermarket product zone, top-down of tiled floor, ambient indoor lighting"
]
# First prompts ever tested
first_prompts = [
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


# First per prompt presets
second_prompts = {
    "avocado": ["An unripe whole avocado, centered on a pure white background, natural lighting, high clarity, professional produce photo for a grocery store"], # from the outside
    "apple": ["A single fresh red apple placed centrally on a white background, studio lighting, no shadows, supermarket product photo"],
    "lemon": ["A ripe yellow lemon centered on a seamless white background, high-resolution studio photo, minimal shadows, product-style image"],
    "banana": ["A banana bundle placed centrally on a plain white background, professional lighting, clear and detailed, supermarket-style photo"],
    "cucumber": ["A fresh whole cucumber in the center of a white background, sharp focus, minimal shadows, clean product photography"],
    "tomato sauce": ["A glass jar of tomato sauce with label facing forward, centered on a white background, bright lighting, realistic packaging style photo"],
    "fruit tea": ["A box of fruit tea with visible branding and design, placed in the center on a white background, sharp studio image, supermarket shelf style"],
    "coffee": ["Studio shot of a supermarket coffee packaging on a white background, centered, clear details, minimal shadows, professional lighting"],
    "coke": ["A single nobrand cola soft drink in the center of a white background, clean studio lighting, sharp focus, typical product image"],
    "spaghetti": ["A spaghetti package from the outside with supermarket label, opaque, placed on a white background, product-centered, supermarket-ready photo"]
    
}


#! Per prompt execution with directions:
DIRECTIONS = [
    "front view", "back view", "top view", "bottom view",
    "left side view", "right side view",
    "angled front-left view", "angled front-right view",
    "angled top-down view", "angled bottom-up view"
]

PER_PROMPT = {
    "avocado": [
        "A whole unripe avocado, %direction%, centered on a pure white background, natural lighting, high clarity, isolated product photo",
        "A whole ripe Hass avocado, dark skin, %direction%, placed alone on white background, sharp focus, minimal shadows, product-style"
    ],
    "apple": [
        "A single fresh red apple, %direction%, centered on a seamless white background, studio lighting, no shadows, product-ready",
        "A lone green Granny Smith apple, %direction%, plain white backdrop, sharp lighting, detailed supermarket-style photo",
        "A golden yellow apple, %direction%, smooth skin, white background, high resolution, isolated product photo"
    ],
    "lemon": [
        "A single whole ripe lemon, %direction%, placed on a white background, soft lighting, minimal shadows, clean produce photo",
        "A halved lemon showing juicy interior, %direction%, isolated on white background, sharp focus, studio-lit",
        "A single Meyer lemon with a smoother skin, %direction%, placed centrally on white, professional grocery photo style"
    ],
    "banana": [
        "Multiple ripe yellow bananas, %direction%, placed flat on a white seamless background, soft shadows, high-resolution photo",
        "A green banana, %direction%, isolated on clean white backdrop, soft studio lighting, sharp clarity, fresh produce image",
        "A ripe banana with light brown spots, %direction%, centered on white background, natural appearance, grocery product style"
    ],
    "cucumber": [
        "A whole fresh cucumber, %direction%, placed centrally on white background, minimal shadows, crisp studio image",
        "A single shrink-wrapped cucumber with barcode label, %direction%, isolated on white, supermarket-style packaging image"
    ],
    "tomato sauce": [
        "A single glass jar of tomato sauce with label facing forward, %direction%, centered on a white background, bright lighting, packaging photo",
        "A pouch of tomato sauce standing upright, %direction%, white background, clear branding, minimal shadows, commercial-style",
        "A cardboard tomato sauce box, closed and upright, %direction%, placed on seamless white background, realistic product image"
    ],
    "fruit tea": [
        "A single fruit tea box with visible branding, %direction%, centered on a pure white background, studio lighting, sharp product photo",
    ],
    "coffee": [
        "A single upright coffee bag with branding, %direction%, centered on a white background, clean studio lighting, isolated packaging photo",
        "A tin of instant coffee with label forward, %direction%, placed centrally on seamless white background, clear studio shot",
        "A soft pack of ground coffee, slightly puffed, %direction%, centered on white, supermarket photo style"
    ],
    "coke": [
        "A plastic bottle of cola, 500ml, no label branding, %direction%, centered on white backdrop, sharp commercial photo",
        "A glass bottle of cola, cap on, %direction%, standing alone on a white background, realistic lighting, isolated product image"
    ],
    "spaghetti": [
        "An opaque spaghetti package with branding, %direction%, centered on a seamless white background, product-style studio lighting",
        "A cardboard box of spaghetti, unopened, %direction%, placed flat on white, minimal shadows, retail packaging photo",
    ]
}

PROMPT_PRESETS = []
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

if PER_PROMPT_MODE:
    with tqdm(total=len(PER_PROMPT) * amount, desc="Overall Progress") as overall_pbar:
        for cla, prompts in PER_PROMPT.items():
            with tqdm(total=len(prompts) * amount, desc=f"Class: {cla}") as class_pbar:
                class_name = cla
                for current_prompt in prompts:
                    prompt["8"]["inputs"]["text"] = current_prompt
                    for direction in DIRECTIONS:
                        current_prompt = current_prompt.replace("%direction%", direction)
                        prompt["3"]["inputs"]["filename_prefix"] = f"{PATH_PREFIX}/{class_name}/{direction.replace(' ', '_')}/{class_name}"
                        print(f"Prompt: {current_prompt}")
                        for i in range(amount):
                            new_seed = random.getrandbits(64)
                            prompt["15"]["inputs"]["seed"] = new_seed
                            prompt["5"]["inputs"]["text"] = NEGATIVE
                            # Queue the prompt for each batch
                            queue_prompt(ws, prompt)
                            class_pbar.update(1)
                            overall_pbar.update(1)
            
else:
    with tqdm(total=len(PROMPT_PRESETS) * len(CLASSES) * amount, desc="Overall Progress") as overall_pbar:
        for cla in CLASSES:
            with tqdm(total=len(PROMPT_PRESETS) * amount, desc=f"Class: {cla}") as class_pbar:
                prompt_nr = 0
                for preset in PROMPT_PRESETS:
                    print(f"Prompt: {preset}")
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
print("All prompts executed successfully.")