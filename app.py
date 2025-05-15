from PIL import Image
import numpy as np
import base64
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from reportlab.lib.utils import ImageReader
from flask import send_file, jsonify, request
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import io, torch, os
from reportlab.lib import colors
from datetime import datetime

os.environ['GOOGLE_API_KEY'] = "AIzaSyCv2dNQMCD3-9s3E5Th7bDy4ko0dyucRCc"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Setup
app = Flask(__name__)
CORS(app)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
# model.load_state_dict(torch.load(r"E:\FYP Work\FYP_code\backend\mask2former-ade-(splicing1_2).pth", map_location=device))
model.load_state_dict(torch.load(r"mask2former-ade-(splicing1_2).pth", map_location=device))
model = model.to(device)
model.eval()

# ========== Flask routes ==========

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Encode original image to base64
        original_image_buffer = io.BytesIO()
        image.save(original_image_buffer, format="PNG")
        original_image_base64 = base64.b64encode(original_image_buffer.getvalue()).decode("utf-8")
            
        # Process image using Mask2Former processor
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Process outputs
        predicted_segmentation = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        
        # Convert to numpy array for visualization
        segmentation_mask = predicted_segmentation.cpu().numpy()
        
        # ========== Create visualizations ==========
        # Create side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Input Image")
        axes[1].imshow(segmentation_mask)
        axes[1].set_title("Prediction")
        
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        # Save visualization to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        visualization_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # ========== Encode mask separately ==========
        # Normalize mask to 0-255 range
        mask_normalized = (segmentation_mask - segmentation_mask.min()) * (255.0 / (segmentation_mask.max() - segmentation_mask.min()))
        mask_image = Image.fromarray(mask_normalized.astype(np.uint8))
        
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")


        #VLM code
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        # Create multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    #"text": "Please explain briefly where the manipulation has been occured, don't use mask"
                     "text": " This is an image and its predicted binary mask showing manipulated regions in white. "
                     "Please explain briefly in 2-3 lines where the manipulation occurred and what might have been altered."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{original_image_base64}"
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{mask_base64}"
                    },
                },
            ]
        )

        # Get response
        response = llm.invoke([message])
        print(response.content)

        return jsonify({
            "original_image": original_image_base64,
            "mask": mask_base64,
            "visualization": visualization_base64,
            "message": response.content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import json
from threading import Lock

counter_file = "counter.json"
counter_lock = Lock()

def get_case_id():
    today = datetime.now().strftime('%Y%m%d')

    with counter_lock:
        if os.path.exists(counter_file):
            with open(counter_file, "r") as f:
                data = json.load(f)
        else:
            data = {}

        count = data.get(today, 0) + 1
        data[today] = count

        with open(counter_file, "w") as f:
            json.dump(data, f)

    return f"DFD-{today}-{count:03d}"


@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # === Process Image ===
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_segmentation = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        segmentation_mask = predicted_segmentation.cpu().numpy()

        # === Create Mask Image ===
        mask_normalized = (segmentation_mask - segmentation_mask.min()) * (255.0 / (segmentation_mask.max() - segmentation_mask.min()))
        mask_image = Image.fromarray(mask_normalized.astype(np.uint8)).convert("L")

        # === Prepare Images ===
        image.save("temp_input.png")
        mask_image.save("temp_mask.png")

        # === Get LLM Analysis ===
        # Encode images for LLM
        original_buffer = io.BytesIO()
        image.save(original_buffer, format="PNG")
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode("utf-8")
        
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")

        # Get professional analysis from Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                     "text": " This is an image and its predicted binary mask showing manipulated regions in white. "
                     "Please explain briefly where the manipulation occurred and what might have been altered."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{original_base64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{mask_base64}"},
                },
            ]
        )
        llm_response = llm.invoke([message]).content

        # === Generate PDF Report ===
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # === Professional Report Design ===
        # Light blue background
        c.setFillColorRGB(0.96, 0.96, 1)
        c.rect(0, 0, width, height, fill=1, stroke=0)

        # Dark blue header
        c.setFillColorRGB(0, 0.2, 0.4)
        c.rect(0, height-80, width, 80, fill=1, stroke=0)
        
        # Title
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width/2, height-50, "DIGITAL IMAGE AUTHENTICITY REPORT")
        c.setFont("Helvetica", 10)
        c.drawCentredString(width/2, height-70, "Forensic Analysis Report")

        # Metadata
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 9)
        c.drawString(40, height-100, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        case_id = get_case_id()
        c.drawString(width-200, height-100, f"Case ID: {case_id}")

        # Divider
        c.setStrokeColorRGB(0, 0.4, 0.6)
        c.setLineWidth(1)
        c.line(40, height-110, width-40, height-110)

        # === Analysis Summary ===
        c.setFillColorRGB(0, 0.3, 0.6)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, height-140, "EXECUTIVE SUMMARY")
        
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 10)
        summary_text = [
            "This report presents forensic analysis of potential digital manipulations",
            "using state-of-the-art AI detection models. Key findings are summarized below."
        ]
        text_object = c.beginText(40, height-160)
        text_object.setFont("Helvetica", 10)
        text_object.setLeading(14)
        for line in summary_text:
            text_object.textLine(line)
        c.drawText(text_object)

        # === Image Evidence ===
        img_y = height-420
        img_width = 220
        img_height = 220
        
        # Original Image
        c.drawImage("temp_input.png", 40, img_y, width=img_width, height=img_height)
        c.setFillColorRGB(0, 0.3, 0.6)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, img_y-20, "ORIGINAL IMAGE")
        
        # Detection Result
        c.drawImage("temp_mask.png", width-260, img_y, width=img_width, height=img_height)
        c.drawString(width-260, img_y-20, "DETECTION HEATMAP")

        # === AI Analysis Section ===
        c.setFillColorRGB(0, 0.3, 0.6)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, img_y-50, "AI FORENSIC ANALYSIS")
        
        # Format LLM response with proper line breaks
        from textwrap import wrap
        analysis_lines = []
        for paragraph in llm_response.split('\n'):
            analysis_lines.extend(wrap(paragraph, width=90))
        
        text_object = c.beginText(40, img_y-70)
        text_object.setFont("Helvetica", 10)
        text_object.setLeading(14)
        
        # Show first 10 lines (adjust based on space)
        for line in analysis_lines[:10]:
            text_object.textLine(line)
        
        if len(analysis_lines) > 10:
            text_object.textLine("\n[Full analysis available in digital report]")
        
        c.drawText(text_object)

        # === Technical Details ===
        c.setFillColorRGB(0, 0.3, 0.6)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, img_y-180, "TECHNICAL SPECIFICATIONS")
        
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 10)
        tech_details = [
            f"Analysis Model: Mask2Former-Swin (ADE20K Fine-tuned)",
            #f"Detection Threshold: {segmentation_mask.max():.2f} confidence",
            f"Processing Date: {datetime.now().strftime('%Y-%m-%d')}",
            "Report Version: 1.1"
        ]
        text_object = c.beginText(40, img_y-200)
        text_object.setFont("Helvetica", 10)
        text_object.setLeading(14)
        for line in tech_details:
            text_object.textLine(line)
        c.drawText(text_object)

        # === Footer ===
        c.setFillColorRGB(0, 0.2, 0.4)
        c.rect(0, 40, width, 40, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica", 8)
        c.drawCentredString(width/2, 65, "This report was generated by AI forensic tools and should be verified by human experts")
        c.drawCentredString(width/2, 55, "Sukkur IBA University | Digital Forensics Lab | © 2024 Deepfake Research Project")

        c.save()
        buffer.seek(0)

        # Cleanup
        os.remove("temp_input.png")
        os.remove("temp_mask.png")

        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"forensic_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)