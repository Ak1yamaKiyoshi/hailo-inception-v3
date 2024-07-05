from typing import List
import os 

os.makedirs("config", exist_ok=True)

contents:str = ""
with open("imagenet_classes.txt", "r+") as f:
    contents = f.read()

to_write = f"""detection_threshold=0.1\n"""

processed:List[str] = contents.split("\n")
for value in processed:
  to_write += f"label={value}\n"

with open("config/inception_v3_config.txt", "w+") as f:
    f.write(to_write)
