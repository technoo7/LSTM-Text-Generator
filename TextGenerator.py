from src.model_handler import load_model, load_tokenizer, load_uti, generate_text
from utils import get_config_value

model = load_model()
tokenizer = load_tokenizer()
uti = load_uti()
text_length = int(get_config_value("config.conf", "Text Generation Settings", "text_length"))
creativity = float(get_config_value("config.conf", "Text Generation Settings", "creativity"))

while True:
    prompt = input("User: ")
    a = generate_text(prompt, text_length, creativity, tokenizer, model, uti)
    if "exit" == prompt:
        break
    