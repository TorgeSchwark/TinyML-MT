import json
from tqdm import tqdm
import uuid
import websocket
import urllib.request
import urllib.parse

# ----- Model settings -----
BATCH_SIZE = 2
IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 20


# ----- Pipeline settings -----
WORKFLOW = "code_imagegeneration/comfyui/sd_api.json"
CLASSES = ["apple", "lemon", "banana", "cucumber", "avocado", "tomato sauce", "fruit tea", "coffee", "spaghetti", "coke"]
AMOUNT_PER_CLASS = 2
PATH_PREFIX = "sd/test"

PROMPT_PRESETS = ["high-resolution photo of a small centered %CLASS% on a white studio background, realistic lighting"]

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
with tqdm(total=len(PROMPT_PRESETS) * len(CLASSES) * amount, desc="Overall Progress") as overall_pbar:
    for cla in CLASSES:
        with tqdm(total=len(PROMPT_PRESETS) * amount, desc=f"Class: {cla}") as class_pbar:
            for preset in PROMPT_PRESETS:
                current_prompt = preset.replace("%CLASS%", cla)
                prompt["8"]["inputs"]["text"] = current_prompt
                prompt["3"]["inputs"]["filename_prefix"] = f"{PATH_PREFIX}/{cla}/{cla}"
                # Queue the prompt for each batch
                for i in range(amount):
                    queue_prompt(ws, prompt)
                    class_pbar.update(1)
                    overall_pbar.update(1)


ws.close()
print("All prompts queued successfully.")