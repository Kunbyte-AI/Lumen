import gradio as gr
import tempfile
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import decord
from PIL import Image
import numpy as np
from diffsynth import ModelManager, WanVideoPipeline, save_video

num_frames, width, height = 49, 832, 480
gpu_id = 0
device = f'cuda:{gpu_id}'

rmbg_model = AutoModelForImageSegmentation.from_pretrained('ckpt/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
rmbg_model.to(device)
rmbg_model.eval()

model_manager = ModelManager(device="cpu") # 1.3b: device=cpu: uses 6G VRAM, device=device: uses 16G VRAM; about 1-2 min per video
wan_dit_path = 'ckpt/Lumen/Lumen-T2V-1.3B-V1.0.ckpt'

if 'wan14b' in wan_dit_path.lower(): # 14B: uses about 36G, about 10 min per video 
    model_manager.load_models(
        [
            wan_dit_path if wan_dit_path else 'ckpt/Wan2.1-Fun-14B-Control/diffusion_pytorch_model.safetensors', 
            'ckpt/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth', 
            'ckpt/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth', 
            'ckpt/Wan2.1-Fun-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth', 
        ], 
        torch_dtype=torch.bfloat16, # float8_e4m3fn fp8量化; bfloat16 
    ) 
else: 
    model_manager.load_models( 
        [ 
            wan_dit_path if wan_dit_path else 'ckpt/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors', 
            'ckpt/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth', 
            'ckpt/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth', 
            'ckpt/Wan2.1-Fun-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth', 
        ], 
        torch_dtype=torch.bfloat16, 
    )
wan_pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
wan_pipe.enable_vram_management(num_persistent_param_in_dit=None)

gr_info_duration = 2 # gradio popup information duration

def rmbg_mask(video_path, mask_path=None, progress=gr.Progress()):
    """Extract foreground from video, return foreground video path"""
    if not video_path:
        gr.Warning("Please upload a video first!", duration=gr_info_duration)
        return None
    
    try:
        progress(0, desc="Preparing foreground extraction...")
    
        if mask_path and os.path.exists(mask_path):
            gr.Info("Using uploaded mask video for foreground extraction.", duration=gr_info_duration)
            
            video_frames = decord.VideoReader(uri=video_path, width=width, height=height)
            video_frames = video_frames.get_batch(range(num_frames)).asnumpy().astype(np.uint8)
            
            mask_frames = decord.VideoReader(uri=mask_path, width=width, height=height)
            mask_frames = mask_frames.get_batch(range(num_frames)).asnumpy().astype(np.uint8)
            
            fg_frames = np.where( mask_frames >= 127, video_frames, 0)
            fg_frames = [Image.fromarray(frame) for frame in fg_frames]

        else:
            image_size = (width, height)
            transform_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
            video_frames = video_reader.get_batch(range(num_frames)).asnumpy()
            fg_frames = []
            
            # Use progress bar in the loop
            for i in range(num_frames):
                # Update progress bar based on processed frames
                progress((i + 1) / num_frames, desc=f"Processing frame {i+1}/{num_frames}...")
                
                image = Image.fromarray(video_frames[i])
                input_images = transform_image(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    preds = rmbg_model(input_images)[-1].sigmoid().cpu()
                pred = preds[0].squeeze()
                pred_pil = transforms.ToPILImage()(pred)
                mask = pred_pil.resize(image.size) # PIL.Image mode=L
                # Extract foreground from image based on mask
                fg_image = Image.composite(image, Image.new('RGB', image.size), mask) # white areas of mask take image1, black areas take image2
                fg_frames.append(fg_image)

        progress(1.0, desc="Saving video...")
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            fg_video_path = temp_file.name
        save_video(fg_frames, fg_video_path, fps=16, quality=5)
        
        progress(1.0, desc="Foreground extraction completed!")
        # gr.Info("Foreground extraction successful!")
        # gr.Video.update(value=fg_video_path, visible=True)

        return fg_video_path
    except Exception as e:
        error_msg = f"Foreground extraction error: {str(e)}"
        gr.Error(error_msg)
        return None


def video_relighting(fg_video_path, prompt, seed=-1, num_inference_steps=50, video_quality=7,
                    progress=gr.Progress()):
    """Relighting the foreground video base on the text """
    if not fg_video_path or not os.path.exists(fg_video_path):
        gr.Warning("Please extract foreground first!", duration = gr_info_duration)
        return None
    if not prompt:
        gr.Warning("Please provide text prompt for relighting!", duration = gr_info_duration)
        return None
    
    try:
        fg_video = decord.VideoReader(uri=fg_video_path, width=width, height=height)
        fg_video = fg_video.get_batch(range(num_frames)).asnumpy().astype('uint8')

        progress(0.1, desc="relighting video...")
        relit_video = wan_pipe(
            prompt=prompt,
            # negative_prompt = 'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards',
            negative_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
            num_inference_steps=num_inference_steps,
            control_video=fg_video,
            height=height, width=width, num_frames=num_frames,
            seed=seed, tiled=True,

            with_clip_feature = True,
            cond_latents2 = None,
            more_config = None,
        )
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            relit_video_path = temp_file.name
        save_video(relit_video, relit_video_path, fps=16, quality=video_quality)
        progress(1.0, desc="Relighting processing completed!")
        gr.Info(f"Relighting successful! Used seed={seed}, steps={num_inference_steps}", duration=gr_info_duration)

        return relit_video_path
    except Exception as e:
        error_msg = f"Relighting processing error: {str(e)}"
        gr.Error(error_msg)
        return None

# gradio app_lumen.py python app_lumen.py
# Examples
bg_prompt_path = 'my_data/zh_short_prompts.txt'
with open(bg_prompt_path, 'r') as f:
    bg_prompts = f.readlines()
bg_prompts = [bg.strip() for bg in bg_prompts if bg.strip()]  # 去除空行
bg_prompts_zh = bg_prompts[ : len(bg_prompts)//2]
bg_prompts_en = bg_prompts[ len(bg_prompts)//2 :]

video_dir = 'test/pachong_test/video/single'
relight_dir = ''


# Create Gradio interface
with gr.Blocks(title="Lumen: Video Relighting Model").queue() as demo:
    gr.Markdown("# 💡Lumen: Consistent Video Relighting and Harmonious Background Replacement\n # <center>with Video Generative Models ([Project Page](https://lumen-relight.github.io/))</center>")
    gr.Markdown('💡 **Lumen** is a video relighting model that can relight the foreground and replace the background of a video base on the input text. The **usage steps** are as follows:')
    gr.Markdown('1. **Upload Video** (will use the first 49 frames and be resized to 832*480). \n' \
    '2. **Extract Foreground**. We use [RMBG2.0](https://github.com/ai-anchorite/BRIA-RMBG-2.0) to extract the foreground but it may get unstable results. If so, we recommend to use [MatAnyone](https://huggingface.co/spaces/PeiqingYang/MatAnyone) to get the **black-and-white mask video**(Alpha Output) and upload it, and then click the **S2** button. \n' \
    '3. **Input Caption**. Select or input the caption you want the video to be. We recommend you to use any LLM ( e.g. [Deepseek](https://chat.deepseek.com/), [Qwen](https://www.tongyi.com/) ) to expand the caption with a simple prompt (请发挥想象力, 扩充下面的视频描述, 如背景, 环境光对前景的影响等), since long prompts may get better results. ' \
    '\n 4. **Relight Video**. ')
    
    # Row 1: video area, using nested layout to achieve 0.4:0.2:0.4 ratio
    with gr.Row():
        # Left area: uploaded video and foreground video
        with gr.Column(scale=3):
            with gr.Row():
                video_input = gr.Video(label="S1. Upload Origin Video") # , scale=0.5
                fg_video = gr.Video(label="Foreground Video or Upload your Mask Video")
        
        # Right area: relit video
        with gr.Column(scale=2):
            relit_video = gr.Video(label="S4. Relighted Video")
    
    # Row 2: two buttons on left and right
    with gr.Row():
        extract_btn = gr.Button("S2. Extract Foreground", variant="secondary", size="md")
        relight_btn = gr.Button("S4. Relight Video (~2 min)", variant="secondary", size="md")
    
    # Row 3: text input box and advanced parameters
    with gr.Row():
        # with gr.Column(scale=3):
        combined_text = gr.Textbox(label="S3. Text Prompt", lines=2, 
            placeholder="Click options below to add captions or fill it with your imagination..."
        )
    
    # Row 4: More settings; can be 
    with gr.Accordion("More Settings", open=False):
        with gr.Row():
            seed = gr.Number(value=-1, minimum=-1, label="Seed", precision=0, info="Set to -1 for random seed (seed>=-1)")
            steps = gr.Number(value=50, minimum=1, label="Inference Steps", precision=0, info="More steps = better result but slower (step>0)")
            video_quality = gr.Number(value=7, minimum=1, maximum=10, label="Video Quality", precision=0, info="The picture quality of the output video (1-10)")

    # Row 5: 将中英文提示合并为tab选项
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("中文描述"):
                    zh_prompts = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        samples=[[text] for text in bg_prompts_zh],
                        label="点击选择视频描述, 多选将叠加",
                        samples_per_page=len(bg_prompts_zh),
                    )
                with gr.Tab("English Prompts"):
                    en_prompts = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        samples=[[text] for text in bg_prompts_en],
                        label="Click to select the video caption",
                        samples_per_page=len(bg_prompts_en),
                    )
        
        # with gr.Column():
        #     gr.Markdown("### Video Relighting Examples of Lumen(1.3B)")
        #     # 准备示例数据
        #     example_inputs = []
        #     for i in range(len(video_names)):
        #         # demo_ori_path, text, demo_res_path
        #         demo_ori_path = os.path.join(video_dir, f"{video_names[i]}.mp4")
        #         text = bg_prompts[i]
        #         demo_res_path = os.path.join(relight_dir, f"{i+1:03d}.mp4")
        #         example_inputs.append([demo_ori_path, text, demo_res_path])
            
        #     # 使用 gr.Examples 组件直接显示视频
        #     gr.Examples(
        #         examples=example_inputs,
        #         inputs=[video_input, combined_text, relit_video],
        #         # cache_examples=True,
        #         label="Click to select an example video and caption. (seed=-1, steps=50, quality=7)",
        #         examples_per_page=len(video_names),
        #     )

    # Set foreground extraction button event - directly call rmbg_mask
    extract_btn.click(
        rmbg_mask,
        inputs=[video_input, fg_video],
        outputs=[fg_video],
    )
    
    # Set relighting button event - directly call video_relighting with new parameters
    relight_btn.click(
        video_relighting,
        inputs=[fg_video, combined_text, seed, steps, video_quality],
        outputs=[relit_video]
    )

    # Add selection event for Dataset component
    def select_option(evt: gr.SelectData, current_text):
        selected_text = evt.value[0]  # Get selected text value
        if not current_text:
            return selected_text
        return f"{current_text}, {selected_text}"
    
    # Bind Dataset selection event
    zh_prompts.select(
        select_option,
        inputs=[combined_text],
        outputs=[combined_text]
    )
    en_prompts.select(
        select_option,
        inputs=[combined_text],
        outputs=[combined_text]
    )


# Launch application
if __name__ == "__main__":
    demo.launch()
    # demo.launch( share=True )


