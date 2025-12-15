import torch
from PIL import Image
import gradio as gr
from gradio_image_annotation import image_annotator
import gc

from src.lora_helper import set_single_lora

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from src.detail_encoder import DetailEncoder
from src.kontext_custom_pipeline import FluxKontextPipelineWithPhotoEncoderAddTokens

hf_hub_download(repo_id="ziheng1234/ImageCritic", filename="detail_encoder.safetensors", local_dir="models")
hf_hub_download(repo_id="ziheng1234/ImageCritic", filename="lora.safetensors", local_dir="models")

from huggingface_hub import snapshot_download

repo_id = "ziheng1234/kontext"
local_dir = "./kontext"
snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model", resume_download=True, max_workers=8)
base_path = "./models"
detail_encoder_path = f"{base_path}/detail_encoder.safetensors"
kontext_lora_path = f"{base_path}/lora.safetensors"


def pick_kontext_resolution(w: int, h: int) -> tuple[int, int]:
    PREFERRED_KONTEXT_RESOLUTIONS = [
        (672, 1568),
        (688, 1504),
        (720, 1456),
        (752, 1392),
        (800, 1328),
        (832, 1248),
        (880, 1184),
        (944, 1104),
        (1024, 1024),
        (1104, 944),
        (1184, 880),
        (1248, 832),
        (1328, 800),
        (1392, 752),
        (1456, 720),
        (1504, 688),
        (1568, 672),
    ]
    target_ratio = w / h
    return min(
        PREFERRED_KONTEXT_RESOLUTIONS,
        key=lambda wh: abs((wh[0] / wh[1]) - target_ratio),
    )


device = None
pipeline = None
transformer = None
detail_encoder = None


def clear_memory():
    """清理显存和内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_models():
    global device, pipeline, transformer, detail_encoder

    if pipeline is not None:
        return

    print("CUDA 可用：", torch.cuda.is_available())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("使用设备：", device)

    # ✅ 优化1: 使用更低精度以节省显存
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("加载 FluxKontextPipelineWithPhotoEncoderAddTokens...")
    pipeline_local = FluxKontextPipelineWithPhotoEncoderAddTokens.from_pretrained(
        "./kontext",
        torch_dtype=dtype,
    )

    # ✅ 优化2: 启用Group-level CPU offloading来减少显存占用
    if torch.cuda.is_available():
        # 使用group-level offload提供更好的性能和灵活性
        onload_device = torch.device("cuda:0")
        offload_device = torch.device("cpu")
        pipeline_local.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            num_blocks_per_group=1,
            use_stream=True,
            record_stream=True,
        )

        # ✅ 优化3: VAE切片以减少峰值显存
        if hasattr(pipeline_local, "vae"):
            pipeline_local.vae.enable_slicing()
            pipeline_local.vae.enable_tiling()

        # ✅ 优化4: 启用attention slicing
        if hasattr(pipeline_local, "enable_attention_slicing"):
            pipeline_local.enable_attention_slicing(1)
    else:
        pipeline_local.to(device)

    print("加载 detail_encoder 权重...")
    state_dict = load_file(detail_encoder_path)
    detail_encoder_local = DetailEncoder().to(dtype=dtype, device=device)

    with torch.no_grad():
        for name, param in detail_encoder_local.named_parameters():
            if name in state_dict:
                added = state_dict[name].to(param.device)
                param.add_(added)

    pipeline_local.detail_encoder = detail_encoder_local

    print("加载 LoRA...")
    set_single_lora(pipeline_local.transformer, kontext_lora_path, lora_weights=[1.0])

    # ✅ 优化5: 加载完成后清理显存
    clear_memory()

    print("模型加载完成！")

    pipeline = pipeline_local
    pipeline.to("cuda")

    detail_encoder = detail_encoder_local


def extract_first_box(annotations: dict):
    """
    从 gradio_image_annotation 的返回中拿第一个 bbox 和对应的 PIL 图像及 patch
    如果没有 bbox，则自动使用整张图作为 bbox。
    """
    if not annotations:
        raise gr.Error("Missing annotation data. Please check if an image is uploaded.")

    img_array = annotations.get("image", None)
    boxes = annotations.get("boxes", [])

    if img_array is None:
        raise gr.Error("No 'image' field found in annotation.")

    img = Image.fromarray(img_array)

    if not boxes:
        w, h = img.size
        xmin, ymin, xmax, ymax = 0, 0, w, h
    else:
        box = boxes[0]
        xmin = int(box["xmin"])
        ymin = int(box["ymin"])
        xmax = int(box["xmax"])
        ymax = int(box["ymax"])

        if xmax <= xmin or ymax <= ymin:
            raise gr.Error("Invalid bbox, please draw the box again.")

    patch = img.crop((xmin, ymin, xmax, ymax))
    return img, patch, (xmin, ymin, xmax, ymax)


def run_with_two_bboxes(
    annotations_A: dict | None,
    annotations_B: dict | None,
    object_name: str,
    base_seed: int = 0,
):
    """生成图像并优化显存使用"""

    load_models()
    global pipeline, device

    if annotations_A is None:
        raise gr.Error("please upload reference image and draw a bbox")
    if annotations_B is None:
        raise gr.Error("please upload input image to be corrected and draw a bbox")

    # ✅ 优化6: 生成前清理显存
    clear_memory()

    img1_full, patch_A, bbox_A = extract_first_box(annotations_A)
    img2_full, patch_B, bbox_B = extract_first_box(annotations_B)

    xmin_B, ymin_B, xmax_B, ymax_B = bbox_B
    patch_w = xmax_B - xmin_B
    patch_h = ymax_B - ymin_B

    if not object_name:
        object_name = "object"

    orig_w, orig_h = patch_B.size
    target_w, target_h = pick_kontext_resolution(orig_w, orig_h)
    width_for_model, height_for_model = target_w, target_h

    cond_A_image = patch_A.resize((width_for_model, height_for_model), Image.Resampling.LANCZOS)
    cond_B_image = patch_B.resize((width_for_model, height_for_model), Image.Resampling.LANCZOS)

    prompt = f"use the {object_name} in IMG1 as a reference to refine, replace, enhance the {object_name} in IMG2"
    print("prompt:", prompt)

    seed = int(base_seed)
    gen_device = device.split(":")[0] if "cuda" in device else device
    generator = torch.Generator(gen_device).manual_seed(seed)

    try:
        # ✅ 优化7: 使用torch.no_grad()减少显存占用
        with torch.no_grad():
            out = pipeline(
                image_A=cond_A_image,
                image_B=cond_B_image,
                prompt=prompt,
                height=height_for_model,
                width=width_for_model,
                guidance_scale=3.5,
                generator=generator,
            )

        gen_patch_model = out.images[0]

        # ✅ 优化8: 生成后立即清理显存
        clear_memory()

        gen_patch = gen_patch_model.resize((patch_w, patch_h), Image.Resampling.LANCZOS)

        composed = img2_full.copy()
        composed.paste(gen_patch, (xmin_B, ymin_B))
        patch_A_resized = patch_A.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
        patch_B_resized = patch_B.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
        SPACING = 10
        collage_w = patch_w * 3 + SPACING * 2
        collage_h = patch_h

        collage = Image.new("RGB", (collage_w, collage_h), (255, 255, 255))

        x0 = 0
        x1 = patch_w + SPACING
        x2 = patch_w * 2 + SPACING * 2

        collage.paste(patch_A_resized, (x0, 0))
        collage.paste(patch_B_resized, (x1, 0))
        collage.paste(gen_patch, (x2, 0))

        return collage, composed

    except Exception as e:
        print(f"生成图像时发生错误: {e}")
        clear_memory()  # 错误时也清理显存
        raise gr.Error(f"生成失败：{str(e)}")


import gradio as gr

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
/* Global Clean Font */

/* Center container */
.app-container {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto;
}

/* Title block */
.title-block h1 {
    text-align: center;
    font-size: 3rem;
    font-weight: 1100;
    background: linear-gradient(90deg, #5b8dff, #b57aff);
    -webkit-background-clip: text;
    color: transparent;
}

.title-block h2 {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.4rem;
    background: linear-gradient(90deg, #6da0ff, #c28aff);
    -webkit-background-clip: text;
    color: transparent;
}

/* Simple card */
.clean-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

/* Card title */
.clean-card-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #404040;
    margin-bottom: 6px;
}

/* Subtitle */
.clean-card-subtitle {
    font-size: 1.1rem;
    color: #404040;
    margin-bottom: 8px;
}

/* Output card */
.output-card {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 14px 16px;
}
.output-card1 {
    background: #ffffff;
    border: none !important;
    box-shadow: none !important;
    border-radius: 12px;
    padding: 14px 16px;
}

/* 渐变主按钮 */
button.color-btn,
.color-btn button {
    width: 100%;
    background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 14px !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
    transition: 0.2s ease !important;
    cursor: pointer;
}

button.color-btn:hover,
.color-btn button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}

button.color-btn:active,
.color-btn button:active {
    transform: scale(0.98) !important;
}

.color-btn > div {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

.example-image img {
    height: 400px !important;
    object-fit: contain;
}
""",
) as demo:
    gen_patch_out = None
    composed_out = None

    gr.Markdown(
        """
    <div class="title-block">
        <h1>The Consistency Critic:</h1>
        <h2>Correcting Inconsistencies in Generated Images via Reference-Guided Attentive Alignment</h2>
    </div>
        """
    )

    gr.Markdown(
        """
    <div class="clean-card">
        <div class="clean-card-title">💡 Tips</div>
        <div class="clean-card-subtitle">
            • Crop both the bbox that needs to be corrected and the reference bbox, preferably covering the smallest repeating unit, to achieve better results.<br>
            • The bbox area should ideally cover the region to be corrected and the reference region as completely as possible.<br>
            • The aspect ratio of the bboxes should also be kept consistent to avoid errors caused by incorrect scaling.<br>
            • If model fails to correct the image, it may be because the generated image is too similar to the reference image, causing the model to skip the repair. You can manually<b> paint that area black on a drawing board before sending to the model, or try cropping only the local region and performing multiple rounds correcting to progressively enhance the whole generated image.</b>
    </div>
        """
    )
    with gr.Row(elem_classes="app-container"):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        <div class="clean-card">
                            <div class="clean-card-title">📌 Reference Image</div>
                            <div class="clean-card-subtitle">Draw a bounding box around the area for reference.</div>
                        </div>
                        """
                    )

                    annotator_A = image_annotator(
                        value=None,
                        label="reference image",
                        label_list=["bbox for reference"],
                        label_colors=[(168, 160, 194)],
                        single_box=True,
                        image_type="numpy",
                        sources=["upload", "clipboard"],
                        height=300,
                    )

                with gr.Column():
                    gr.Markdown(
                        """
                        <div class="clean-card">
                            <div class="clean-card-title">🖼️ Input Image To Be Corrected</div>
                            <div class="clean-card-subtitle">Use the mouse wheel to zoom and draw a bounding box around the area to be corrected.</div>
                        </div>
                        """
                    )

                    annotator_B = image_annotator(
                        value=None,
                        label="input image to be corrected",
                        label_list=["bbox for correction"],
                        label_colors=[(168, 160, 194)],
                        single_box=True,
                        image_type="numpy",
                        sources=["upload", "clipboard"],
                        height=300,
                    )

            with gr.Row():
                object_name = gr.Textbox(
                    label="Caption for object (optional; using 'product' also works)",
                    value="product",
                    placeholder="e.g. product, shoes, bag, face ...",
                )

                base_seed = gr.Number(
                    label="Seed",
                    value=0,
                    precision=0,
                )

            with gr.Row():
                run_btn = gr.Button("✨ Generate ", elem_classes="color-btn")

        with gr.Column():
            with gr.Column(elem_classes="output-card1"):
                gen_patch_out = gr.Image(label="concatenated input-output", interactive=False)

            with gr.Column(elem_classes="output-card1"):
                composed_out = gr.Image(label="corrected image", interactive=False)

    with gr.Column(elem_classes="clean-card"):
        gr.Markdown(
            """
            <div style="
                font-size: 1.3rem;
                font-weight: 600;
                color: #404040;
                margin-bottom: 6px;
            ">
                📚 Example Images
            </div>
            """,
        )

        gr.Markdown(
            """
            <div style="
                font-size: 1.1rem;
                color: #404040;
                margin-bottom: 8px;
            ">
                Below are some example pairs showing how bounding boxes should be drawn.
                You can click and drag the image below into the upper area for generation.<br>
               <b> Full-image input is also supported, but it is recommended to  use the smallest possible bounding box that covers the region to be corrected and reference bbox. For example, the bbox approach used in the first row generally produces better results than the one used in the second row.</b> 
            </div>
            """,
        )
        with gr.Row():
            gr.Image("./test_imgs/product_3.png", label="reference example", elem_classes="example-image")
            gr.Image("./test_imgs/product_3_bbox_1.png", label="reference example with bbox", elem_classes="example-image")
            gr.Image("./test_imgs/generated_3.png", label="input example", elem_classes="example-image")
            gr.Image("./test_imgs/generated_3_bbox_1.png", label="input example with bbox", elem_classes="example-image")

        with gr.Row():
            gr.Image("./test_imgs/product_3.png", label="reference example", elem_classes="example-image")
            gr.Image("./test_imgs/product_3_bbox.png", label="reference example with bbox", elem_classes="example-image")
            gr.Image("./test_imgs/generated_3.png", label="input example", elem_classes="example-image")
            gr.Image("./test_imgs/generated_3_bbox.png", label="input example with bbox", elem_classes="example-image")

        with gr.Row():
            gr.Image("./test_imgs/product_1.jpg", label="reference example", elem_classes="example-image")
            gr.Image("./test_imgs/product_1_bbox.png", label="reference example with bbox", elem_classes="example-image")
            gr.Image("./test_imgs/generated_1.png", label="input example", elem_classes="example-image")
            gr.Image("./test_imgs/generated_1_bbox.png", label="input example with bbox", elem_classes="example-image")

        with gr.Row():
            gr.Image("./test_imgs/product_2.png", label="reference example", elem_classes="example-image")
            gr.Image("./test_imgs/product_2_bbox.png", label="reference example with bbox", elem_classes="example-image")
            gr.Image("./test_imgs/generated_2.png", label="input example", elem_classes="example-image")
            gr.Image("./test_imgs/generated_2_bbox.png", label="input example with bbox", elem_classes="example-image")

    run_btn.click(
        fn=run_with_two_bboxes,
        inputs=[annotator_A, annotator_B, object_name, base_seed],
        outputs=[gen_patch_out, composed_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7779)
