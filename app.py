import cv2
import gradio as gr
from src.pipeline.avis_pipeline import AVISPipeline
from src.utils.logger import logger

# Initialize the pipeline in lazy mode so the UI boots instantly.
# The heavy models (BLIP, EfficientNet) will load when the first snapshot is taken.
pipeline = AVISPipeline(lazy_load=True)

def process_image(image_input):
    if image_input is None:
        return None, "Please capture or upload an image first.", ""
    
    try:
        # Gradio provides the image as a NumPy RGB array.
        # Our pipeline (cv2) expects BGR format.
        bgr_image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        
        # Run inference
        result = pipeline.run_image(bgr_image, annotate=True)
        
        # Convert annotated result back to RGB for Gradio to display
        if result.annotated_frame is not None:
            out_image = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
        else:
            out_image = image_input
            
        # Format the output text
        agentic_cap = result.agentic_caption
        
        faces_info = "\n".join([f"- **Face {i+1}**: {f.emotion.label.title()} ({f.emotion.confidence*100:.1f}%)" for i, f in enumerate(result.faces)])
        if not faces_info:
            faces_info = "- No faces detected"
            
        stats = f"### ⏱️ Processing Time: {result.processing_time_ms:.1f} ms\n\n### 🎭 Detected Emotions:\n{faces_info}"
        
        return out_image, agentic_cap, stats
        
    except Exception as e:
        logger.error(f"Error processing image in UI: {e}")
        return image_input, f"❌ Error: {str(e)}", ""

# Define the Cyber/Dark Mode Aesthetic using Gradio Themes
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=custom_theme, title="AVIS") as app:
    with gr.Row():
        gr.Markdown(
            """
            # 👁️ AVIS — Affective Visual Intelligence System
            **A Multi-Agent Architecture for Emotion-Aware Contextual Image Captioning**
            """
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("📷 Webcam Snapshot"):
                    gr.Markdown("Strike a pose and click the **Camera Icon** 📷 inside the video feed below to capture and process instantly!")
                    webcam_input = gr.Image(sources=["webcam"], type="numpy", label="Webcam", streaming=False)
                
                with gr.Tab("🖼️ Image Upload"):
                    gr.Markdown("Upload an image containing faces.")
                    upload_input = gr.Image(sources=["upload"], type="numpy", label="Upload Image")
                    upload_btn = gr.Button("🧠 Analyze Image", variant="primary", size="lg")
                    
        with gr.Column(scale=1):
            output_image = gr.Image(type="numpy", label="Annotated Result", interactive=False)
            agentic_output = gr.Textbox(label="🤖 AVIS Agentic Caption", lines=4, interactive=False, text_align="left")
            stats_output = gr.Markdown(label="System Stats")

    # Connect UI events to the backend function
    webcam_input.change(
        fn=process_image,
        inputs=[webcam_input],
        outputs=[output_image, agentic_output, stats_output]
    )
    
    upload_btn.click(
        fn=process_image,
        inputs=[upload_input],
        outputs=[output_image, agentic_output, stats_output]
    )

if __name__ == "__main__":
    logger.info("Starting Gradio Web Server on http://localhost:7860")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
