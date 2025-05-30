from doctr.models import ocr_predictor
from nltk.corpus import words, stopwords
from nltk.metrics import edit_distance
import tkinter as tk
from tkinter import messagebox, Label, filedialog, StringVar, Text
from PIL import Image, ImageEnhance, ImageFilter, ImageTk
import os
from doctr.io import DocumentFile
from nltk import pos_tag, word_tokenize, FreqDist
from symspellpy import SymSpell, Verbosity
import importlib
import threading
import subprocess

# Pure Python implementation for cleaning words (no C++/ctypes needed)
def clean_word_py(word):
    return ''.join(char for char in word if char.isalpha())

# Add model options (only doctr and transformer models, with ratings)
# Format: (Display Name, Model Key, Speed Rating, Accuracy Rating)
MODEL_OPTIONS = [
    ("DocTR CRNN+VGG16 [S:8 A:7]", "doctr_crnn_vgg16", 8, 7),
    ("DocTR SAR [S:8 A:8]", "doctr_sar", 8, 8),
    ("DocTR PARSeq [S:7 A:8]", "doctr_parseq", 7, 8),
    ("TrOCR (Base, no downtraining) [S:6 A:9]", "trocr_base", 6, 9),
    ("TrOCR (Large, no downtraining) [S:4 A:9]", "trocr_large", 4, 9),
    ("Donut [S:5 A:8]", "donut", 5, 8),
    ("EMNIST CNN [S:9 A:6]", "emnist_cnn", 9, 6),
]

def load_ocr_model(model_key):
    if model_key == "doctr_crnn_vgg16":
        return ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
    elif model_key == "doctr_sar":
        return ocr_predictor(det_arch="db_resnet50", reco_arch="sar_resnet31", pretrained=True)
    elif model_key == "doctr_parseq":
        return ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True)
    elif model_key == "trocr_base":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        return ("trocr", processor, model)
    elif model_key == "trocr_large":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        return ("trocr", processor, model)
    elif model_key == "donut":
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        return ("donut", processor, model)
    elif model_key == "emnist_cnn":
        import torch
        import torch.nn as nn
        class SimpleEMNISTCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 62)  # <-- Change to 62 for EMNIST ByClass
            def forward(self, x):
                x = nn.functional.relu(self.conv1(x))
                x = nn.functional.relu(self.conv2(x))
                x = nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = nn.functional.relu(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)
                output = nn.functional.log_softmax(x, dim=1)
                return output
        model_path = os.path.join(os.path.dirname(__file__), "emnist_cnn (1).pt")
        emnist_labels = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z'
        ]
        model = SimpleEMNISTCNN()
        state_dict = torch.load(model_path, map_location="cpu")
        # Remove any unexpected keys if present
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return (model_key, model, emnist_labels)
    # Fallback to doctr_crnn_vgg16
    return ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)

# Lazy model loading: don't load any model at startup
selected_model_key = None
ocr_model = None
ocr_model_type = None

def ensure_model_loaded():
    global ocr_model, ocr_model_type, selected_model_key
    # Always reload the model when the selection changes to avoid stale model state
    selected = model_var.get()
    for name, key, *_ in MODEL_OPTIONS:
        if name == selected:
            selected_model_key = key
            break
    if selected_model_key is None:
        selected_model_key = MODEL_OPTIONS[0][1]
    ocr_model = load_ocr_model(selected_model_key)
    ocr_model_type = selected_model_key.split("_")[0] if "_" in selected_model_key else selected_model_key

def download_model(model_key):
    progress_var.set(f"Downloading model: {model_key} ...")
    root.update()
    try:
        if model_key == "trocr_base":
            subprocess.check_call(["pip", "install", "transformers", "torch"])
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        elif model_key == "trocr_large":
            subprocess.check_call(["pip", "install", "transformers", "torch"])
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        elif model_key == "donut":
            subprocess.check_call(["pip", "install", "transformers", "torch"])
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        elif model_key == "doctr_crnn_vgg16":
            subprocess.check_call(["pip", "install", "python-doctr[torch]"])
        elif model_key == "doctr_sar":
            subprocess.check_call(["pip", "install", "python-doctr[torch]"])
        elif model_key == "doctr_parseq":
            subprocess.check_call(["pip", "install", "python-doctr[torch]"])
        # EMNIST CNN and others assumed to be present
        progress_var.set(f"Model {model_key} downloaded.")
    except Exception as e:
        progress_var.set(f"Download failed: {e}")
    root.update()

def ensure_model_downloaded(model_key):
    # Download in a thread and show progress
    def _download():
        progress_var.set(f"Checking and downloading model: {model_key} ...")
        root.update()
        download_model(model_key)
        progress_var.set(f"Model {model_key} ready.")
        root.update()
    threading.Thread(target=_download, daemon=True).start()

def on_model_change(event=None):
    # Always reload the model immediately on change
    ensure_model_loaded()
    selected = model_var.get()
    progress_var.set(f"Model set to: {selected}")
    root.update()

english_words = set(words.words())
stop_words = set(stopwords.words('english'))

# Initialize SymSpell for English
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# Load dictionary (frequency dictionary included with symspellpy)
dict_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
if os.path.exists(dict_path):
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
else:
    # Fallback: use a small in-memory dictionary from nltk words
    for w in english_words:
        sym_spell.create_dictionary_entry(w, 1)

def preprocess_image(image_path, times=1, output_folder="Process Output"):
    os.makedirs(output_folder, exist_ok=True)
    image = Image.open(image_path).convert("L")
    preprocessed_path = None
    for i in range(1, times + 1):
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        image = image.convert("L")
        preprocessed_path = os.path.join(
            output_folder,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_preprocessed_{i}_{threading.get_ident()}.png"
        )
        image.save(preprocessed_path)
    print(f"[DEBUG] Preprocessed image saved as: {preprocessed_path}")
    return preprocessed_path

def recognize_text_from_image(image_path):
    ensure_model_loaded()
    print(f"[DEBUG] OCR model: {ocr_model}, Model key: {selected_model_key}, Image path: {image_path}")
    # Use the selected model for OCR
    if isinstance(ocr_model, tuple) and ocr_model[0] == "trocr":
        from PIL import Image as PILImage
        processor, model = ocr_model[1], ocr_model[2]
        image = PILImage.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"[DEBUG] TrOCR output: {text}")
        return text.strip()
    elif isinstance(ocr_model, tuple) and ocr_model[0] == "donut":
        from PIL import Image as PILImage
        processor, model = ocr_model[1], ocr_model[2]
        image = PILImage.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_length=512)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"[DEBUG] Donut output: {text}")
        return text.strip()
    elif isinstance(ocr_model, tuple) and ocr_model[0] == "emnist_cnn":
        from PIL import Image as PILImage
        import torch
        import torchvision.transforms as transforms
        import numpy as np

        model = ocr_model[1]
        emnist_labels = ocr_model[2]
        image = PILImage.open(image_path).convert("L")
        img_np = np.array(image)
        projection = np.sum(img_np < 128, axis=0)
        threshold = max(projection) * 0.2 if np.max(projection) > 0 else 1
        in_char = False
        char_bounds = []
        start = 0
        for i, val in enumerate(projection):
            if val > threshold and not in_char:
                in_char = True
                start = i
            elif val <= threshold and in_char:
                in_char = False
                end = i
                if end - start > 2:
                    char_bounds.append((start, end))
        if in_char and (len(projection) - start > 2):
            char_bounds.append((start, len(projection)))

        chars = []
        for (start, end) in char_bounds:
            char_img = image.crop((start, 0, end, image.height)).resize((28, 28))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            input_tensor = transform(char_img).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1, keepdim=True).item()
            chars.append(emnist_labels[pred] if pred < len(emnist_labels) else str(pred))
        result = "".join(chars)
        print(f"[DEBUG] EMNIST output (segmented): {result}")
        return result
    else:
        doc = DocumentFile.from_images(image_path)
        result = ocr_model(doc)
        exported_result = result.export()
        text = []
        if "pages" in exported_result:
            for page in exported_result["pages"]:
                if "blocks" in page:
                    for block in page["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                if "words" in line:
                                    text.extend(word["value"] for word in line["words"])
        joined = ' '.join(text).strip()
        print(f"[DEBUG] DocTR output: {joined}")
        return joined

def extract_keywords(text, top_n=5):
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.isalpha() and w.lower() not in stop_words]
    tagged = pos_tag(filtered)
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    freq = FreqDist(nouns)
    return [word for word, _ in freq.most_common(top_n)]

def correct_ocr_text(text):
    words_list = text.split()
    corrected_words = []
    for word in words_list:
        clean = clean_word_py(word)
        if not clean:
            corrected_words.append(word)
            continue
        suggestions = sym_spell.lookup(clean, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            best = suggestions[0].term
            corrected_words.append(best.capitalize() if word.istitle() else best)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

def spellcheck_text(text):
    return correct_ocr_text(text)

def process_single_image(image_path, preprocess_times=1, correction_times=1):
    # Always reload the model before processing to ensure correct model is used
    ensure_model_loaded()
    print(f"[DEBUG] process_single_image: {image_path}, preprocess_times={preprocess_times}")
    do_preprocess = preprocess_var.get()
    if do_preprocess:
        progress_var.set("Preprocessing image...")
        root.update()
        preprocessed_image_path = preprocess_image(image_path, times=preprocess_times)
    else:
        preprocessed_image_path = image_path
    progress_var.set("Running OCR...")
    root.update()
    print(f"[DEBUG] OCR on: {preprocessed_image_path}")
    recognized_text = recognize_text_from_image(preprocessed_image_path)
    ocr_text_box.config(state="normal")
    ocr_text_box.delete("1.0", tk.END)
    ocr_text_box.insert(tk.END, recognized_text)
    ocr_text_box.config(state="normal")
    progress_var.set("Done.")
    root.update()
    print(f"[DEBUG] Final OCR text: {recognized_text}")
    return recognized_text

def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*")]
    )
    if not file_path:
        return
    print(f"[DEBUG] Selected image: {file_path}")
    img = Image.open(file_path)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    image_label.update()
    ocr_text_box.config(state="normal")
    ocr_text_box.delete("1.0", tk.END)
    ocr_text_box.config(state="normal")
    try:
        times = int(preprocess_entry.get())
    except Exception:
        times = 1
    process_single_image(file_path, preprocess_times=times, correction_times=1)

def start_processing():
    ensure_model_loaded()
    try:
        times = int(preprocess_entry.get())
    except Exception:
        times = 1
    data_folder = "data"
    try:
        process_images_in_folder(data_folder, preprocess_times=times, correction_times=1, gui_update=update_image)
        messagebox.showinfo("Done", "Processing completed.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def update_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    image_label.update()

def process_images_in_folder(folder_path, preprocess_times=1, correction_times=1, gui_update=None):
    output_folder = "recognised_text"
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for idx, file_name in enumerate(files, 1):
        image_path = os.path.join(folder_path, file_name)
        if gui_update:
            gui_update(image_path)
        preprocessed_image_path = preprocess_image(image_path, times=preprocess_times)
        recognized_text = recognize_text_from_image(preprocessed_image_path)
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            formatted_text = "\n".join([recognized_text[i:i+80] for i in range(0, len(recognized_text), 80)])
            f.write(formatted_text)
        progress_var.set(f"Processed {idx}/{len(files)}: {file_name}")
        root.update()
    progress_var.set("Batch processing done.")
    root.update()

def download_model(model_key):
    pass

def ensure_model_downloaded(model_key):
    pass

def ask_and_download_model():
    pass

def on_model_change(event=None):
    global ocr_model, ocr_model_type, selected_model_key
    selected = model_var.get()
    for name, key, *_ in MODEL_OPTIONS:
        if name == selected:
            selected_model_key = key
            break

    def reload_model():
        progress_var.set(f"Loading model: {selected} ...")
        root.update()
        try:
            model = load_ocr_model(selected_model_key)
            globals()["ocr_model"] = model
            globals()["ocr_model_type"] = selected_model_key.split("_")[0] if "_" in selected_model_key else selected_model_key
            progress_var.set(f"Model set to: {selected}")
        except Exception as e:
            progress_var.set(f"Failed to load model: {e}")
        root.update()

    threading.Thread(target=reload_model, daemon=True).start()

# ...existing code...

root = tk.Tk()
root.title("OCR Image Processing")

# 0th column: Options
tk.Label(root, text="Preprocess Times:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
preprocess_entry = tk.Entry(root)
preprocess_entry.insert(0, "1")
preprocess_entry.grid(row=1, column=0, padx=10, pady=5)
preprocess_var = tk.BooleanVar(value=True)
preprocess_check = tk.Checkbutton(root, text="Enable Preprocessing", variable=preprocess_var)
preprocess_check.grid(row=2, column=0, padx=10, pady=2, sticky="w")

# Model selection dropdown with ratings
tk.Label(root, text="OCR Model:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
model_var = tk.StringVar(value=MODEL_OPTIONS[0][0])
model_menu = tk.OptionMenu(root, model_var, *(name for name, *_ in MODEL_OPTIONS), command=on_model_change)
model_menu.grid(row=4, column=0, padx=10, pady=5)

select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.grid(row=5, column=0, padx=10, pady=5)
process_button = tk.Button(root, text="Batch Folder (Optional)", command=start_processing)
process_button.grid(row=6, column=0, padx=10, pady=5)

# 1st column: Image
image_label = Label(root)
image_label.grid(row=0, column=1, rowspan=8, padx=10, pady=10)

# 2nd column: Progress, OCR Text
progress_var = StringVar()
progress_label = tk.Label(root, textvariable=progress_var, fg="blue")
progress_label.grid(row=0, column=2, padx=10, pady=5, sticky="w")

tk.Label(root, text="OCR Text:").grid(row=1, column=2, padx=10, pady=5, sticky="nw")
ocr_text_box = Text(root, wrap="word", height=20, width=50)
ocr_text_box.grid(row=2, column=2,rowspan=6, padx=10, pady=5, sticky="w")
ocr_text_box.config(state="normal")

root.mainloop()