# import os
# import shutil
# import subprocess
# from dotenv import load_dotenv

# load_dotenv()

# ollama_path = os.environ.get("OLLAMA_PATH", "ollama")

# print("OLLAMA_PATH:", ollama_path)
# print("Exists on disk:", os.path.exists(ollama_path))
# print("shutil.which():", shutil.which(ollama_path))

# try:
#     result = subprocess.run(
#         [ollama_path, "--version"],
#         capture_output=True,
#         text=True,
#         timeout=10,
#     )

#     print("\nReturn code:", result.returncode)
#     print("STDOUT:")
#     print(result.stdout)

#     print("STDERR:")
#     print(result.stderr)

# except Exception as e:
#     print("\nERROR:")
#     print(type(e).__name__)
#     print(e)

import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

ollama_path = os.environ["OLLAMA_PATH"]

result = subprocess.run(
    [ollama_path, "run", "qwen3:4b", "Say hello"],
    capture_output=True,
    text=True,
    timeout=60,
)

print("Return code:", result.returncode)
print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)