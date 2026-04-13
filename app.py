import os
import sys
import tempfile
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import load_model, predict

model = load_model()

def analyse(image):
    if image is None:
        return "No image provided", 0.0

    # Save PIL image to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict(model, tmp_path)
        probability = float(result)
    finally:
        os.unlink(tmp_path)

    pct = round(probability * 100, 1)

    if probability >= 0.5:
        label = f"🚨 DMC Action Required — {pct}% spillover probability"
    else:
        label = f"✅ No Action Needed — {pct}% spillover probability"

    return label, probability

demo = gr.Interface(
    fn=analyse,
    inputs=gr.Image(type="pil", label="Upload or Capture Image"),
    outputs=[
        gr.Text(label="Result"),
        gr.Slider(minimum=0, maximum=1, label="Spillover Probability", interactive=False)
    ],
    title="DMC — Dustbin Management Committee",
    description="Upload a dustbin image to detect garbage spillover.",
    allow_flagging="never"
)

demo.launch()
