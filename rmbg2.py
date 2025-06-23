from PIL import Image
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import json

def process_videos_on_gpu(gpu_id, video_paths_chunk, num_frames, width, height):
    from transformers import AutoModelForImageSegmentation
    import torch
    from torchvision import transforms
    from pathlib import Path
    import decord
    from diffsynth import save_video

    image_size = (width, height)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = AutoModelForImageSegmentation.from_pretrained('ckpt/RMBG-2.0', trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    device = f'cuda:{gpu_id}'
    model.to(device)
    model.eval()

    for video_path in tqdm(video_paths_chunk, desc=f'GPU {gpu_id}'):
        masks_save_path = video_path.replace('/video/', '/video_rmbg_msk/')
        # masks_save_path = video_path.replace('/video_len49/', '/video_len49_rmbg_msk/')

        os.makedirs(os.path.dirname(masks_save_path), exist_ok=True)
        # 建议提前遍历文件夹, 创建好路径, 多线程创建文件夹可能存在冲突

        try: # 测试mask是否存在
            assert os.path.exists(masks_save_path) == True 
            # 如果存在, 尝试加载
            if isinstance(masks_save_path, str): masks_save_path = Path(masks_save_path)
            mask_reader = decord.VideoReader(uri=masks_save_path.as_posix(), width=width, height=height)
        except:
            if isinstance(video_path, str): video_path = Path(video_path)
            video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
            video_frames = video_reader.get_batch(range(num_frames)).asnumpy()
            mask_frames = []
            for i in range(num_frames):
                image = Image.fromarray(video_frames[i])
                input_images = transform_image(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    preds = model(input_images)[-1].sigmoid().cpu()
                pred = preds[0].squeeze()
                pred_pil = transforms.ToPILImage()(pred)
                mask = pred_pil.resize(image.size)
                mask_frames.append(mask)
            save_video(mask_frames, masks_save_path, fps=16, quality=5)

if __name__ == '__main__': # python rmbg2.py
    # with open('my_data/....json', 'r') as f:
        # video_paths = json.load(f)
    # video_paths = [os.path.join('my_data', path) for path in video_paths]

    vide_dir = 'test/pachong_test/video'
    video_paths = []
    for root, dirs, files in os.walk(vide_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
    
    print('len(video_paths):', len(video_paths))

    num_frames, width, height = 49, 832, 480
    # num_frames, width, height = 16, 512, 512 # for lav

    gpu_ids = [ 4,5,6,7 ]
    num_workers_per_gpu = 1

    num_gpus = len(gpu_ids)
    chunk_size = (len(video_paths) + num_gpus - 1) // num_gpus
    video_chunks = [video_paths[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]

    mp.set_start_method('spawn', force=True)
    processes = []
    for gpu_idx, (gpu_id, chunk) in enumerate(zip(gpu_ids, video_chunks)):
        if not chunk:
            continue
        # 将每张卡上的视频再分成num_workers_per_gpu份
        worker_chunk_size = (len(chunk) + num_workers_per_gpu - 1) // num_workers_per_gpu
        for worker_idx in range(num_workers_per_gpu):
            start = worker_idx * worker_chunk_size
            end = min((worker_idx + 1) * worker_chunk_size, len(chunk))
            sub_chunk = chunk[start:end]
            if not sub_chunk:
                continue
            p = mp.Process(target=process_videos_on_gpu, args=(gpu_id, sub_chunk, num_frames, width, height))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
