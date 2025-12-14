import requests
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os
import tempfile
import json
import hashlib

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not available. Audio processing will be limited.")
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Video processing will be limited.")
try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Latent processing will be limited.")


# 全局任务状态管理器，用于跟踪所有异步任务
class AsyncTaskManager:
    """管理所有异步任务的状态"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AsyncTaskManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.tasks = {}  # task_id -> task_info dict
            self.tasks_lock = threading.Lock()
            self._initialized = True
    
    def register_task(self, task_id, api_key, base_url, is_webapp_task=False):
        """注册一个新任务"""
        with self.tasks_lock:
            if task_id not in self.tasks:
                self.tasks[task_id] = {
                    "api_key": api_key,
                    "base_url": base_url,
                    "is_webapp_task": is_webapp_task,
                    "status": "PENDING",  # PENDING, RUNNING, COMPLETED, ERROR
                    "result": None,
                    "error": None,
                    "created_at": time.time()
                }
                print(f"Registered async task: {task_id}")
            return self.tasks[task_id]
    
    def get_task(self, task_id):
        """获取任务信息"""
        with self.tasks_lock:
            return self.tasks.get(task_id)
    
    def update_task_status(self, task_id, status, result=None, error=None):
        """更新任务状态"""
        with self.tasks_lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = status
                if result is not None:
                    self.tasks[task_id]["result"] = result
                if error is not None:
                    self.tasks[task_id]["error"] = error
                print(f"Updated task {task_id} status to {status}")
    
    def get_task_result(self, task_id):
        """获取已完成任务的结果（如果已缓存）"""
        with self.tasks_lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                if task_info["status"] == "COMPLETED" and task_info.get("result") is not None:
                    return task_info["result"]
            return None
    
    def remove_task(self, task_id):
        """移除任务（可选，用于清理）"""
        with self.tasks_lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                print(f"Removed task: {task_id}")
    
    def remove_tasks(self, task_ids):
        """批量移除任务"""
        with self.tasks_lock:
            removed_count = 0
            for task_id in task_ids:
                if task_id in self.tasks:
                    del self.tasks[task_id]
                    removed_count += 1
            if removed_count > 0:
                print(f"Removed {removed_count} tasks from task manager: {task_ids}")
            return removed_count
    
    def get_all_task_ids(self):
        """获取所有任务ID列表"""
        with self.tasks_lock:
            return list(self.tasks.keys())


class AsyncBatchExecuteNode:
    """
    异步批量执行节点 - 合并了 AsyncExecute、AsyncCollect 和 AsyncBatchResult 的功能
    1. 创建多个异步任务（最多20个）- 并发执行
    2. 收集所有 task_id
    3. 批量等待并获取所有任务的结果 - 并发查询和处理
    """
    
    def __init__(self):
        self.task_manager = AsyncTaskManager()
    
    def _create_task_http(self, apiConfig, nodeInfoList, is_webapp_task):
        """独立的 HTTP 请求方法：创建任务"""
        api_key = apiConfig.get("apiKey")
        base_url = apiConfig.get("base_url")
        
        # 优先从 nodeInfoList 中提取 workflowId_webappId（在过滤之前提取）
        retrieved_workflow_id = self._extract_workflowId_webappId(nodeInfoList)
        # 如果 nodeInfoList 中没有，则从 apiConfig 中获取（向后兼容）
        if not retrieved_workflow_id:
            retrieved_workflow_id = apiConfig.get("workflowId_webappId")
        
        # 过滤 nodeInfoList，移除元数据项（nodeId=-1 和 nodeId=-2）
        filtered_nodeInfoList = self._filter_node_info_list(nodeInfoList)
        
        if not api_key or not base_url:
            raise ValueError("Missing required apiConfig fields: apiKey, base_url")
        
        if is_webapp_task:
            if not retrieved_workflow_id:
                raise ValueError("workflowId_webappId (acting as webappId) must be provided when is_webapp_task is True. Please provide it in WebAppTaskConfigNode or apiConfig.")
            url = f"{base_url}/task/openapi/ai-app/run"
            try:
                webappId_int = int(retrieved_workflow_id)
            except (ValueError, TypeError):
                raise ValueError(f"webappId must be a valid integer, got: {retrieved_workflow_id}")
            data = {
                "webappId": webappId_int,
                "apiKey": api_key,
                "nodeInfoList": filtered_nodeInfoList or [],
            }
        else:
            if not retrieved_workflow_id:
                raise ValueError("workflowId_webappId must be provided for standard ComfyUI tasks. Please provide it in WebAppTaskConfigNode or apiConfig.")
            url = f"{base_url}/task/openapi/create"
            data = {
                "workflowId": retrieved_workflow_id,
                "apiKey": api_key,
                "nodeInfoList": filtered_nodeInfoList or [],
            }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
        }
        
        max_retries = 5
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                if result.get("code") == 0:
                    task_data = result.get("data", {})
                    task_id = task_data.get("taskId")
                    if task_id:
                        return task_id
                    else:
                        raise ValueError("Missing taskId in task creation response.")
                else:
                    api_msg = result.get('msg', 'Unknown API error')
                    raise Exception(f"API error (code {result.get('code')}): {api_msg}")
                    
            except (requests.exceptions.RequestException, json.JSONDecodeError, Exception) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise Exception(f"Failed to create task after {max_retries} attempts: {last_exception}") from last_exception
        
        raise Exception(f"Failed to create task after {max_retries} attempts: {last_exception}")
    
    def _check_task_status_http(self, task_id, api_key, base_url):
        """独立的 HTTP 请求方法：查询任务状态"""
        url = f"{base_url}/task/openapi/outputs"
        headers = {
            "User-Agent": "ComfyUI-RH-APICall-Node/1.0",
            "Content-Type": "application/json",
        }
        data = {"taskId": task_id, "apiKey": api_key}
        
        max_retries = 5
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, headers=headers, timeout=20)
                
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    if response.status_code != 200:
                        raise requests.exceptions.RequestException(f"HTTP Error {response.status_code} with Invalid JSON")
                    else:
                        return {"taskStatus": "error", "error": "Invalid JSON response"}
                
                api_code = result.get("code")
                api_msg = result.get("msg", "")
                api_data = result.get("data")
                
                if response.status_code != 200:
                    error_detail = api_msg if api_msg else f"HTTP Error {response.status_code}"
                    if 500 <= response.status_code < 600:
                        raise requests.exceptions.RequestException(f"Server Error {response.status_code}: {error_detail}")
                    else:
                        return {"taskStatus": "error", "error": error_detail}
                
                # 成功完成，返回输出数据
                if api_code == 0 and isinstance(api_data, list) and api_data:
                    return api_data
                
                # 任务排队中
                elif api_msg == "APIKEY_TASK_IS_QUEUED":
                    return {"taskStatus": "QUEUED"}
                
                # 任务运行中
                elif api_msg == "APIKEY_TASK_IS_RUNNING":
                    possible_wss_url = None
                    if isinstance(api_data, dict):
                        possible_wss_url = api_data.get("netWssUrl")
                    if possible_wss_url:
                        return {"taskStatus": "RUNNING", "netWssUrl": possible_wss_url}
                    else:
                        return {"taskStatus": "RUNNING"}
                
                # API 错误
                elif api_code != 0:
                    return {"taskStatus": "error", "error": api_msg}
                
                # 任务完成但没有输出
                elif api_code == 0 and isinstance(api_data, list) and not api_data:
                    return {"taskStatus": "completed_no_output"}
                
                # 任务运行中（code 0 但没有数据）
                elif api_code == 0 and api_data is None:
                    return {"taskStatus": "RUNNING"}
                
                # 未知状态
                else:
                    return {"taskStatus": "unknown", "details": result}
                    
            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {"taskStatus": "error", "error": f"Network Error after retries: {last_exception}"}
        
        return {"taskStatus": "error", "error": f"Status check failed after {max_retries} attempts: {last_exception}"}
    
    def _download_image(self, url):
        """下载图片并转换为 tensor"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            image = image.convert("RGB")
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W, 3]
            return img_tensor
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None
    
    def _download_video(self, url):
        """下载视频并提取帧"""
        if not CV2_AVAILABLE:
            print("cv2 not available, cannot process video")
            return []
        
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_path = temp_file.name
            temp_file.close()
            
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            cap = cv2.VideoCapture(temp_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_array = frame_rgb.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_array).unsqueeze(0)  # [1, H, W, 3]
                frames.append(frame_tensor)
            cap.release()
            
            os.remove(temp_path)
            return frames
        except Exception as e:
            print(f"Error processing video from {url}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return []
    
    def _download_and_load_latent(self, url):
        """
        下载并加载 latent 文件
        使用 safetensors 加载，应用 multiplier，返回 {"samples": tensor} 格式
        """
        if not SAFETENSORS_AVAILABLE:
            print("Error: safetensors not available, cannot load latent file")
            return None
        
        max_retries = 5
        retry_delay = 1
        last_exception = None
        latent_path = None
        output_dir = "temp"  # Use temp directory
        
        # Ensure temp directory exists
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Error creating temporary directory {output_dir}: {e}")
                return None
        
        # --- Download the latent file ---
        for attempt in range(max_retries):
            latent_path = None  # Reset path for each attempt
            try:
                # Generate a unique temporary filename
                try:
                    safe_filename = f"temp_latent_{os.path.basename(url)}_{str(int(time.time()*1000))}.latent"
                    safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in safe_filename)[:150]
                    latent_path = os.path.join(output_dir, safe_filename)
                except Exception as path_e:
                    print(f"Error creating temporary latent path: {path_e}")
                    latent_path = os.path.join(output_dir, f"temp_latent_{str(int(time.time()*1000))}.latent")
                
                print(f"Attempt {attempt + 1}/{max_retries} to download latent to temp path: {latent_path}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                downloaded_size = 0
                with open(latent_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                
                if downloaded_size > 0:
                    print(f"Temporary latent downloaded successfully: {latent_path}")
                    break  # Exit retry loop on successful download
                else:
                    print(f"Warning: Downloaded latent file is empty: {latent_path}")
                    if os.path.exists(latent_path):
                        try:
                            os.remove(latent_path)
                        except OSError:
                            pass
                    last_exception = IOError("Downloaded latent file is empty.")
                    # Continue retry loop
                    
            except (requests.exceptions.RequestException, IOError) as e:
                print(f"Download latent attempt {attempt + 1} failed: {e}")
                last_exception = e
                if latent_path and os.path.exists(latent_path):
                    try:
                        os.remove(latent_path)
                    except OSError:
                        pass
                # Continue retry loop
            
            if attempt < max_retries - 1:
                print(f"Retrying download in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
        
        # Check if download succeeded
        if not latent_path or not os.path.exists(latent_path) or os.path.getsize(latent_path) == 0:
            print(f"Failed to download latent {url} successfully after {max_retries} attempts.")
            if latent_path and os.path.exists(latent_path):
                try:
                    os.remove(latent_path)
                except OSError:
                    pass
            return None
        
        # --- Load the latent file ---
        loaded_latent_dict = None
        try:
            print(f"Loading latent from {latent_path}...")
            # Use safetensors.torch.load_file
            latent_content = safetensors.torch.load_file(latent_path, device="cpu")
            
            if "latent_tensor" not in latent_content:
                raise ValueError("'latent_tensor' key not found in the loaded latent file.")
            
            # Apply multiplier based on LoadLatent logic
            multiplier = 1.0
            if "latent_format_version_0" not in latent_content:
                multiplier = 1.0 / 0.18215
                print(f"Applying multiplier {multiplier:.5f} (old latent format detected)")
            
            samples_tensor = latent_content["latent_tensor"].float() * multiplier
            loaded_latent_dict = {"samples": samples_tensor}
            print("Latent loaded successfully.")
            
        except Exception as e:
            print(f"Error loading latent file {latent_path}: {e}")
            # Ensure loaded_latent_dict remains None on error
            loaded_latent_dict = None
        finally:
            # --- Cleanup ---
            if latent_path and os.path.exists(latent_path):
                try:
                    os.remove(latent_path)
                    print(f"Deleted temporary latent file: {latent_path}")
                except OSError as e:
                    print(f"Error deleting temporary latent file {latent_path}: {e}")
        
        return loaded_latent_dict
    
    def _create_placeholder_latent(self, batch_size=1, channels=4, height=64, width=64):
        """创建占位符 latent tensor 字典（与 ExecuteNode 保持一致）"""
        latent = torch.zeros([batch_size, channels, height, width])
        return {"samples": latent}
    
    def _download_and_read_text(self, url):
        """下载并读取文本文件"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error reading text from {url}: {e}")
            return None
    
    def _download_audio(self, url):
        """下载并加载音频文件"""
        if not TORCHAUDIO_AVAILABLE:
            print("torchaudio not available, cannot process audio")
            return None
        
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            waveform, sample_rate = torchaudio.load(temp_path)
            waveform = waveform.to(torch.float32)
            if not waveform.is_contiguous():
                waveform = waveform.contiguous()
            waveform = waveform.unsqueeze(0)  # Add batch dimension
            
            os.remove(temp_path)
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"Error processing audio from {url}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return None
    
    def _process_completed_task_output(self, status_result):
        """处理已完成任务的输出，下载并处理所有文件（不等待，直接处理）"""
        if not isinstance(status_result, list) or len(status_result) == 0:
            raise Exception("Invalid status_result: expected non-empty list")
        
        image_urls = []
        video_urls = []
        latent_urls = []
        text_urls = []
        audio_urls = []
        
        for output in status_result:
            if isinstance(output, dict):
                file_url = output.get("fileUrl")
                file_type = output.get("fileType")
                if file_url and file_type:
                    file_type_lower = file_type.lower()
                    if file_type_lower in ["png", "jpg", "jpeg", "webp", "bmp", "gif"]:
                        image_urls.append(file_url)
                    elif file_type_lower in ["mp4", "avi", "mov", "webm"]:
                        video_urls.append(file_url)
                    elif file_type_lower == "latent":
                        latent_urls.append(file_url)
                    elif file_type_lower == "txt":
                        text_urls.append(file_url)
                    elif file_type_lower in ["wav", "mp3", "flac", "ogg"]:
                        audio_urls.append(file_url)
        
        # 并发下载和处理文件
        image_data_list = []
        frame_data_list = []
        latent_data = None
        text_data = None
        audio_data = None
        
        # 并发下载图片
        if image_urls:
            with ThreadPoolExecutor(max_workers=min(len(image_urls), 10)) as executor:
                futures = {executor.submit(self._download_image, url): url for url in image_urls}
                for future in as_completed(futures):
                    img_tensor = future.result()
                    if img_tensor is not None:
                        image_data_list.append(img_tensor)
        
        # 处理多张图片的归一化（与 ExecuteNode 相同的逻辑）
        if len(image_data_list) > 1:
            max_channels = 0
            max_h = 0
            max_w = 0
            for img in image_data_list:
                max_h = max(max_h, img.shape[1])
                max_w = max(max_w, img.shape[2])
                max_channels = max(max_channels, img.shape[3])
            
            normalized_images = []
            for img_tensor in image_data_list:
                _, h, w, c = img_tensor.shape
                current_img = img_tensor
                
                if c < max_channels:
                    if c == 3 and max_channels == 4:
                        alpha_channel = torch.ones(1, h, w, 1, dtype=current_img.dtype)
                        current_img = torch.cat([current_img, alpha_channel], dim=3)
                    else:
                        padding_channels = max_channels - c
                        padding = torch.zeros(1, h, w, padding_channels, dtype=current_img.dtype)
                        current_img = torch.cat([current_img, padding], dim=3)
                
                if h < max_h or w < max_w:
                    pad_h_total = max_h - h
                    pad_w_total = max_w - w
                    pad_top = pad_h_total // 2
                    pad_bottom = pad_h_total - pad_top
                    pad_left = pad_w_total // 2
                    pad_right = pad_w_total - pad_left
                    
                    img_permuted = current_img.permute(0, 3, 1, 2)
                    padded_permuted = F.pad(img_permuted, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
                    padded_img = padded_permuted.permute(0, 2, 3, 1)
                    normalized_images.append(padded_img)
                else:
                    normalized_images.append(current_img)
            
            image_data_list = normalized_images
        
        # 并发下载视频
        if video_urls:
            with ThreadPoolExecutor(max_workers=min(len(video_urls), 5)) as executor:
                futures = {executor.submit(self._download_video, url): url for url in video_urls}
                for future in as_completed(futures):
                    frames = future.result()
                    if frames:
                        frame_data_list.extend(frames)
        
        # 下载 latent（只取第一个）
        if latent_urls and latent_data is None:
            latent_data = self._download_and_load_latent(latent_urls[0])
        
        # 如果没有 latent，创建占位符（与 ExecuteNode 保持一致）
        if latent_data is None:
            print("No latent generated, creating placeholder.")
            latent_data = self._create_placeholder_latent()
        
        # 下载文本（只取第一个）
        if text_urls and text_data is None:
            text_data = self._download_and_read_text(text_urls[0])
        
        # 下载音频（只取第一个）
        if audio_urls and audio_data is None:
            audio_data = self._download_audio(audio_urls[0])
        
        # 合并图片
        final_images = None
        if image_data_list:
            if len(image_data_list) == 1:
                final_images = image_data_list[0]
            else:
                try:
                    final_images = torch.cat(image_data_list, dim=0)
                except Exception as e:
                    print(f"Warning: Could not merge images: {e}")
                    final_images = image_data_list[0]
        
        # 合并视频帧
        final_video_frames = None
        if frame_data_list:
            if len(frame_data_list) == 1:
                final_video_frames = frame_data_list[0]
            else:
                try:
                    final_video_frames = torch.cat(frame_data_list, dim=0)
                except Exception as e:
                    print(f"Warning: Could not merge video frames: {e}")
                    final_video_frames = frame_data_list[0]
        
        return (final_images, final_video_frames, latent_data, text_data, audio_data)
    
    def _process_task_output(self, task_id, api_key, base_url):
        """处理任务输出，下载并处理所有文件"""
        max_retries = 30
        retry_interval = 1
        max_retry_interval = 5
        consecutive_empty_results = 0
        max_consecutive_empty = 3
        
        for attempt in range(max_retries):
            try:
                status_result = self._check_task_status_http(task_id, api_key, base_url)
                
                # 处理完成但没有输出的情况
                if isinstance(status_result, dict) and status_result.get("taskStatus") == "completed_no_output":
                    raise Exception("Task completed successfully but the workflow produced no output results.")
                
                # 任务仍在运行
                if isinstance(status_result, dict) and status_result.get("taskStatus") in ["RUNNING", "QUEUED"]:
                    if status_result.get("taskStatus") == "RUNNING":
                        consecutive_empty_results += 1
                        if consecutive_empty_results >= max_consecutive_empty:
                            raise Exception("Task completed successfully but the workflow produced no output results.")
                    else:
                        consecutive_empty_results = 0
                    
                    wait_time = min(retry_interval * (1.5 ** attempt), max_retry_interval)
                    time.sleep(wait_time)
                    continue
                
                # 任务完成，有输出
                if isinstance(status_result, list) and len(status_result) > 0:
                    consecutive_empty_results = 0
                    return self._process_completed_task_output(status_result)
                
                # 错误状态
                if isinstance(status_result, dict) and status_result.get("taskStatus") == "error":
                    error_msg = status_result.get('error', 'Unknown error')
                    raise Exception(f"Task failed: {error_msg}")
                
                # 其他未知状态，继续等待
                time.sleep(retry_interval)
                continue
                
            except Exception as e:
                if "completed successfully but" in str(e) or "failed" in str(e).lower():
                    raise
                # 其他错误，继续重试
                time.sleep(retry_interval)
                continue
        
        raise Exception(f"Task output processing timeout after {max_retries} attempts")
    
    def _extract_is_webapp_task(self, nodeInfoList):
        """
        从 nodeInfoList 中提取 is_webapp_task 配置
        查找 nodeId=-1, fieldName="_is_webapp_task" 的特殊标记
        """
        if not nodeInfoList:
            return False
        
        # 处理嵌套列表格式（ComfyUI 的 ARRAY 类型）
        actual_list = nodeInfoList
        if isinstance(nodeInfoList, list) and len(nodeInfoList) > 0:
            if isinstance(nodeInfoList[0], list):
                actual_list = nodeInfoList[0]
        
        # 查找 _is_webapp_task 标记
        for item in actual_list:
            if isinstance(item, dict):
                if item.get("nodeId") == -1 and item.get("fieldName") == "_is_webapp_task":
                    field_value = item.get("fieldValue", "false")
                    return field_value.lower() == "true"
        
        # 默认返回 False
        return False
    
    def _extract_workflowId_webappId(self, nodeInfoList):
        """
        从 nodeInfoList 中提取 workflowId_webappId 配置
        查找 nodeId=-2, fieldName="_workflowId_webappId" 的特殊标记
        如果找到则返回该值，否则返回 None
        """
        if not nodeInfoList:
            return None
        
        # 处理嵌套列表格式（ComfyUI 的 ARRAY 类型）
        actual_list = nodeInfoList
        if isinstance(nodeInfoList, list) and len(nodeInfoList) > 0:
            if isinstance(nodeInfoList[0], list):
                actual_list = nodeInfoList[0]
        
        # 查找 _workflowId_webappId 标记
        for item in actual_list:
            if isinstance(item, dict):
                if item.get("nodeId") == -2 and item.get("fieldName") == "_workflowId_webappId":
                    field_value = item.get("fieldValue", "")
                    if field_value:
                        return field_value
        
        # 默认返回 None
        return None
    
    def _filter_node_info_list(self, nodeInfoList):
        """
        从 nodeInfoList 中过滤掉元数据项（nodeId=-1 和 nodeId=-2 的项）
        只保留实际的工作流节点配置
        """
        if not nodeInfoList:
            return []
        
        # 处理嵌套列表格式（ComfyUI 的 ARRAY 类型）
        actual_list = nodeInfoList
        if isinstance(nodeInfoList, list) and len(nodeInfoList) > 0:
            if isinstance(nodeInfoList[0], list):
                actual_list = nodeInfoList[0]
        
        # 过滤掉元数据项（nodeId=-1 和 nodeId=-2 的项）
        filtered_list = [
            item for item in actual_list
            if isinstance(item, dict) and item.get("nodeId") not in [-1, -2]
        ]
        
        return filtered_list
    
    @classmethod
    def INPUT_TYPES(cls):
        # 创建统一的 apiConfig 和最多50个 task 输入
        inputs = {
            "required": {
                "apiConfig": ("STRUCT", {"tooltip": "所有任务共享的API配置"}),
            },
            "optional": {
                "task_1": ("ARRAY", {"default": [], "tooltip": "第1个任务的节点信息列表（可包含 is_webapp_task 配置）"}),
                "concurrency_limit": ("INT", {"default": 50, "min": 1, "max": 100, "tooltip": "并发限制（默认50）"}),
                "run_timeout": ("INT", {"default": 600, "min": 1, "max": 9999999, "tooltip": "任务超时时间（秒）"}),
                "poll_interval": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 60.0, "step": 0.5, "tooltip": "轮询间隔（秒）"}),
                "wait_all": ("BOOLEAN", {"default": True, "tooltip": "是否等待所有任务完成"}),
                "fail_on_error": ("BOOLEAN", {"default": False, "tooltip": "任何任务失败时是否抛出异常"}),
            }
        }
        
        # 添加最多50个可选的 task 输入（从第2个开始）
        for i in range(2, 51):
            inputs["optional"][f"task_{i}"] = ("ARRAY", {"default": [], "tooltip": f"第{i}个任务的节点信息列表（可包含 is_webapp_task 配置）"})
        
        return inputs
    
    RETURN_TYPES = ("STRUCT",)
    RETURN_NAMES = ("results",)
    
    CATEGORY = "RunningHub"
    FUNCTION = "process"
    OUTPUT_NODE = True
    
    def _create_task(self, apiConfig, nodeInfoList, is_webapp_task):
        """创建单个异步任务，返回 task_id"""
        task_id = self._create_task_http(apiConfig, nodeInfoList, is_webapp_task)
        
        # 注册任务到管理器
        api_key = apiConfig.get("apiKey")
        base_url = apiConfig.get("base_url")
        self.task_manager.register_task(task_id, api_key, base_url, is_webapp_task)
        
        print(f"Async task created successfully. Task ID: {task_id}")
        return task_id    

    @classmethod
    def IS_CHANGED(
        cls,
        apiConfig,
        task_1=None,
        concurrency_limit=50,
        run_timeout=600,
        poll_interval=2.0,
        wait_all=True,
        fail_on_error=False,
        **kwargs
    ):
        """
        只要返回值发生变化，ComfyUI 就会重新执行该节点
        """

        # 收集所有 task_x（task_1 已显式给出，其余在 kwargs 里）
        tasks = []
        if task_1:
            tasks.append(task_1)

        for i in range(2, 51):
            t = kwargs.get(f"task_{i}")
            if t:
                tasks.append(t)

        # 构造一个“决定性输入快照”
        state = {
            "apiConfig": apiConfig,
            "tasks": tasks,
            "concurrency_limit": concurrency_limit,
            "run_timeout": run_timeout,
            "poll_interval": poll_interval,
            "wait_all": wait_all,
            "fail_on_error": fail_on_error,
        }

        # 稳定序列化（防止 dict 顺序问题）
        state_str = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)

        # 生成 hash
        return hashlib.md5(state_str.encode("utf-8")).hexdigest()

    def process(self, apiConfig, task_1=None, concurrency_limit=50, 
                run_timeout=600, poll_interval=2.0, wait_all=True, fail_on_error=False, **kwargs):
        """
        批量执行异步任务：
        1. 创建多个异步任务（所有任务共享 apiConfig，每个任务使用自己的 nodeInfoList）
        2. 从每个 nodeInfoList 中提取 is_webapp_task 配置
        3. 收集所有 task_id
        4. 批量等待并获取所有任务的结果
        """
        try:
            # 步骤1: 收集所有 task，所有任务共享同一个 apiConfig
            task_configs = []
            
            # 第一个任务
            if task_1 is not None:
                # 从 nodeInfoList 中提取 is_webapp_task
                is_webapp_task = self._extract_is_webapp_task(task_1)
                # 注意：不过滤 nodeInfoList，过滤将在 _create_task_http 中进行
                # 这样可以确保 workflowId_webappId 可以在 _create_task_http 中正确提取
                
                task_configs.append({
                    "apiConfig": apiConfig,
                    "nodeInfoList": task_1,  # 传递原始 nodeInfoList，在 _create_task_http 中过滤
                    "is_webapp_task": is_webapp_task
                })
            
            # 收集其他任务（最多50个），所有任务共享同一个 apiConfig，每个任务使用自己的 nodeInfoList
            for i in range(2, 51):
                task_key = f"task_{i}"
                
                if task_key in kwargs and kwargs[task_key] is not None:
                    nodeInfoList = kwargs[task_key]
                    # 从 nodeInfoList 中提取 is_webapp_task
                    is_webapp_task = self._extract_is_webapp_task(nodeInfoList)
                    # 注意：不过滤 nodeInfoList，过滤将在 _create_task_http 中进行
                    # 这样可以确保 workflowId_webappId 可以在 _create_task_http 中正确提取
                    
                    task_configs.append({
                        "apiConfig": apiConfig,  # 所有任务共享同一个 apiConfig
                        "nodeInfoList": nodeInfoList,  # 传递原始 nodeInfoList，在 _create_task_http 中过滤
                        "is_webapp_task": is_webapp_task  # 从 nodeInfoList 中提取
                    })
            
            if not task_configs:
                print("Warning: At least one task is required. Returning empty results.")
                return ([],)
            
            print(f"Creating {len(task_configs)} async tasks concurrently...")
            
            # 步骤2: 并发创建所有任务并收集 task_id
            # 保存 task_id 到索引的映射关系（索引是 1-based，对应 task_1, task_2, ...）
            task_id_to_index = {}  # task_id -> index (1-based)
            task_index_to_id = {}  # index (1-based) -> task_id
            task_creation_errors = {}  # index -> error message (用于创建任务时的错误)
            
            # 使用线程池并发创建任务
            with ThreadPoolExecutor(max_workers=min(len(task_configs), concurrency_limit)) as executor:
                future_to_config = {
                    executor.submit(
                        self._create_task,
                        config["apiConfig"],
                        config["nodeInfoList"],
                        config["is_webapp_task"]
                    ): (i, config) for i, config in enumerate(task_configs, 1)
                }
                
                for future in as_completed(future_to_config):
                    i, config = future_to_config[future]
                    try:
                        task_id = future.result()
                        task_id_to_index[task_id] = i
                        task_index_to_id[i] = task_id
                        print(f"Task {i}/{len(task_configs)} created: {task_id}")
                    except Exception as e:
                        error_msg = f"Failed to create task {i}: {e}"
                        print(error_msg)
                        task_creation_errors[i] = error_msg
                        if fail_on_error:
                            # 即使 fail_on_error=True，也返回包含错误信息的结果而不是抛出异常
                            print(f"Warning: fail_on_error is True, but returning error result instead of raising exception.")
                            # 继续处理，不抛出异常
            
            task_ids = list(task_id_to_index.keys())
            
            if not task_ids:
                if task_creation_errors:
                    error_summary = "; ".join([f"Task {i}: {err}" for i, err in task_creation_errors.items()])
                    print(f"Warning: No tasks were created successfully. Errors: {error_summary}. Returning empty results.")
                else:
                    print("Warning: No tasks were created successfully. Returning empty results.")
                # 返回包含错误信息的结果列表
                results_list = []
                for index, error_msg in task_creation_errors.items():
                    results_list.append({
                        "index": index,
                        "task_id": None,
                        "status": "ERROR",
                        "error": error_msg
                    })
                return (results_list,)
            
            print(f"Successfully created {len(task_ids)}/{len(task_configs)} tasks: {task_ids}")
            
            # 步骤3: 批量等待并获取所有任务的结果
            print(f"Batch processing {len(task_ids)} tasks: {task_ids}")
            
            # 跟踪任务状态
            task_results = {}  # task_id -> result tuple
            task_execution_errors = {}   # task_id -> error message (用于任务执行时的错误)
            completed_tasks = set()
            failed_tasks = set()
            
            batch_start_time = time.time()
            
            # 主循环：等待所有任务完成
            while True:
                # 检查超时
                if time.time() - batch_start_time > run_timeout:
                    incomplete = [tid for tid in task_ids if tid not in completed_tasks and tid not in failed_tasks]
                    error_msg = f"Timeout: {len(incomplete)} tasks did not complete within {run_timeout} seconds. Incomplete tasks: {incomplete}"
                    print(error_msg)
                    # 将超时的任务标记为错误并缓存
                    for task_id in incomplete:
                        timeout_error = f"Task timeout after {run_timeout} seconds"
                        task_execution_errors[task_id] = timeout_error
                        failed_tasks.add(task_id)
                        # 更新任务管理器状态
                        if self.task_manager.get_task(task_id):
                            self.task_manager.update_task_status(task_id, "ERROR", error=timeout_error)
                    # 不再抛出异常，直接返回已完成任务的结果
                    print("Warning: Timeout occurred. Returning results for completed tasks.")
                    break
                
                # 检查所有任务是否完成
                if wait_all:
                    if len(completed_tasks) + len(failed_tasks) >= len(task_ids):
                        print(f"All {len(task_ids)} tasks completed. Successful: {len(completed_tasks)}, Failed: {len(failed_tasks)}")
                        break
                else:
                    # 如果不等待所有任务，只要有任务完成就可以返回
                    if len(completed_tasks) > 0:
                        print(f"{len(completed_tasks)} tasks completed. Returning results...")
                        break
                
                # 并发轮询所有未完成的任务
                pending_tasks = [tid for tid in task_ids if tid not in completed_tasks and tid not in failed_tasks]
                
                if pending_tasks:
                    # 使用线程池并发查询任务状态
                    with ThreadPoolExecutor(max_workers=min(len(pending_tasks), concurrency_limit)) as executor:
                        future_to_task = {}
                        for task_id in pending_tasks:
                            task_info = self.task_manager.get_task(task_id)
                            if not task_info:
                                print(f"Warning: Task {task_id} not found in task manager. Skipping...")
                                continue
                            
                            api_key = task_info.get("api_key")
                            base_url = task_info.get("base_url")
                            
                            if not api_key or not base_url:
                                print(f"Warning: Task {task_id} missing api_key or base_url. Skipping...")
                                continue
                            
                            future = executor.submit(self._check_task_status_http, task_id, api_key, base_url)
                            future_to_task[future] = task_id
                        
                        # 收集已完成的任务，准备并发处理输出
                        completed_status_results = {}  # task_id -> status_result (list)
                        
                        for future in as_completed(future_to_task):
                            task_id = future_to_task[future]
                            try:
                                status_result = future.result()
                                
                                # 检查任务状态
                                if isinstance(status_result, list) and len(status_result) > 0:
                                    # 任务完成，有输出 - 收集起来，稍后并发处理
                                    print(f"Task {task_id} completed successfully. Will process output concurrently...")
                                    completed_status_results[task_id] = status_result
                                
                                elif isinstance(status_result, dict):
                                    task_status = status_result.get("taskStatus")
                                    
                                    if task_status == "error":
                                        error_msg = status_result.get('error', 'Unknown error')
                                        print(f"Task {task_id} failed: {error_msg}")
                                        task_execution_errors[task_id] = error_msg
                                        failed_tasks.add(task_id)
                                        if self.task_manager.get_task(task_id):
                                            self.task_manager.update_task_status(task_id, "ERROR", error=error_msg)
                                        # 不再抛出异常，继续处理其他任务
                                    
                                    elif task_status == "completed_no_output":
                                        error_msg = "Task completed successfully but the workflow produced no output results."
                                        print(f"Task {task_id}: {error_msg}")
                                        task_execution_errors[task_id] = error_msg
                                        failed_tasks.add(task_id)
                                        if self.task_manager.get_task(task_id):
                                            self.task_manager.update_task_status(task_id, "ERROR", error=error_msg)
                                        # 不再抛出异常，继续处理其他任务
                                    
                                    # RUNNING 和 QUEUED 状态不需要特殊处理，继续等待
                            
                            except Exception as e:
                                # 网络错误等，记录但不立即失败
                                if isinstance(e, (TimeoutError, Exception)) and "failed" in str(e).lower():
                                    # 如果是明确的失败错误，标记为失败并缓存
                                    error_msg = str(e)
                                    task_execution_errors[task_id] = error_msg
                                    failed_tasks.add(task_id)
                                    # 更新任务管理器状态
                                    if self.task_manager.get_task(task_id):
                                        self.task_manager.update_task_status(task_id, "ERROR", error=error_msg)
                                    # 不再抛出异常，继续处理其他任务
                                # 其他错误（如网络问题）继续重试
                                print(f"Error checking task {task_id}: {e}. Will retry...")
                        
                        # 并发处理所有已完成任务的输出
                        if completed_status_results:
                            print(f"Concurrently processing outputs for {len(completed_status_results)} completed tasks...")
                            with ThreadPoolExecutor(max_workers=min(len(completed_status_results), concurrency_limit)) as output_executor:
                                output_future_to_task = {
                                    output_executor.submit(self._process_completed_task_output, status_result): task_id
                                    for task_id, status_result in completed_status_results.items()
                                }
                                
                                for output_future in as_completed(output_future_to_task):
                                    task_id = output_future_to_task[output_future]
                                    try:
                                        result = output_future.result()
                                        task_results[task_id] = result
                                        completed_tasks.add(task_id)
                                        # 缓存结果到任务管理器，供 AsyncExtractResultNode 使用
                                        if self.task_manager.get_task(task_id):
                                            self.task_manager.update_task_status(task_id, "COMPLETED", result=result)
                                        print(f"Task {task_id} output processed successfully.")
                                    except Exception as e:
                                        error_msg = f"Error processing output for task {task_id}: {e}"
                                        print(error_msg)
                                        task_execution_errors[task_id] = error_msg
                                        failed_tasks.add(task_id)
                                        if self.task_manager.get_task(task_id):
                                            self.task_manager.update_task_status(task_id, "ERROR", error=error_msg)
                                        # 不再抛出异常，继续处理其他任务
                
                # 等待一段时间后再次轮询
                if len(completed_tasks) + len(failed_tasks) < len(task_ids):
                    time.sleep(poll_interval)
            
            # 组织结果：按照 nodeInfoList 的索引顺序（1-based）排列
            if not task_results and not task_execution_errors and not task_creation_errors:
                print("Warning: No tasks completed successfully. Returning empty results.")
                return ([],)
            
            print(f"Organizing results from {len(task_results)} successful tasks...")
            
            # 按照索引顺序（1-based）组织结果列表
            # results 是一个列表，每个元素对应一个任务的结果（按 task_1, task_2, ... 的顺序）
            results_list = []
            max_index = max(task_index_to_id.keys()) if task_index_to_id else 0
            # 如果只有创建错误，也要包含这些索引
            if task_creation_errors:
                max_index = max(max_index, max(task_creation_errors.keys()) if task_creation_errors else 0)
            
            for index in range(1, max_index + 1):
                if index in task_creation_errors:
                    # 任务创建失败
                    results_list.append({
                        "index": index,
                        "task_id": None,
                        "status": "ERROR",
                        "error": task_creation_errors[index]
                    })
                elif index in task_index_to_id:
                    task_id = task_index_to_id[index]
                    if task_id in task_results:
                        # 成功完成的任务，添加结果
                        result = task_results[task_id]
                        results_list.append({
                            "index": index,
                            "task_id": task_id,
                            "status": "COMPLETED",
                            "result": result  # (images, video_frames, latent, text, audio)
                        })
                    elif task_id in task_execution_errors:
                        # 失败的任务，添加错误信息
                        results_list.append({
                            "index": index,
                            "task_id": task_id,
                            "status": "ERROR",
                            "error": task_execution_errors[task_id]
                        })
                    else:
                        # 未完成的任务
                        results_list.append({
                            "index": index,
                            "task_id": task_id,
                            "status": "PENDING",
                            "error": "Task not completed"
                        })
            
            # 输出警告信息
            total_errors = len(task_creation_errors) + len(task_execution_errors)
            if total_errors > 0:
                print(f"Warning: {total_errors} tasks failed: {len(task_creation_errors)} creation errors, {len(task_execution_errors)} execution errors")
            
            print(f"Batch processing complete. Successfully processed {len(task_results)}/{len(task_ids)} tasks.")
            
            # 返回结果前，任务已缓存到任务管理器（在 process_task_output 时已缓存）
            processed_task_ids = list(completed_tasks) + list(failed_tasks)
            if processed_task_ids:
                print(f"Tasks processed and cached in task manager. Total cached: {len(processed_task_ids)} tasks.")
            
            # 返回 results 列表
            return (results_list,)
        except Exception as e:
            print(f"Error in AsyncBatchExecuteNode: {e}. Returning empty results.")
            import traceback
            traceback.print_exc()
            return ([],)


class AsyncExtractResultNode:
    """
    异步结果提取节点 - 从 AsyncBatchExecuteNode 的 results 中提取特定任务的结果
    通过 task_index 参数指定要提取的任务索引（对应 task_1, task_2, ...）
    """
    
    def __init__(self):
        pass
    
    def _create_placeholder_latent(self, batch_size=1, channels=4, height=64, width=64):
        """创建占位符 latent tensor 字典（与 ExecuteNode 保持一致）"""
        latent = torch.zeros([batch_size, channels, height, width])
        return {"samples": latent}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results": ("STRUCT", {"tooltip": "从 AsyncBatchExecuteNode 输出的 results"}),
                "task_index": ("INT", {"default": 1, "min": 1, "max": 50, "tooltip": "任务索引，对应 task_ 的索引（1-based，即 task_1 对应 1）"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT", "STRING", "AUDIO")
    RETURN_NAMES = ("images", "video_frames", "latent", "text", "audio")
    
    CATEGORY = "RunningHub"
    FUNCTION = "process"
    OUTPUT_NODE = True
    
    def process(self, results, task_index):
        """
        从 results 列表中根据 task_index 提取特定任务的结果
        results: 从 AsyncBatchExecuteNode 输出的结果列表
        task_index: 任务索引（1-based），对应 task_1, task_2, ...
        """
        # 创建占位符 latent（避免返回 None 导致 VAE decode 错误）
        placeholder_latent = self._create_placeholder_latent()
        # 默认返回值（使用占位符 latent 而不是 None）
        default_return = (None, None, placeholder_latent, "", None)
        
        try:
            if not results or not isinstance(results, list):
                print(f"Warning: results must be a non-empty list from AsyncBatchExecuteNode. Returning default values.")
                return default_return
            
            if task_index < 1:
                print(f"Warning: task_index must be >= 1, got {task_index}. Returning default values.")
                return default_return
            
            # 查找对应索引的任务结果
            task_result = None
            for result_item in results:
                if isinstance(result_item, dict) and result_item.get("index") == task_index:
                    task_result = result_item
                    break
            
            if task_result is None:
                available_indices = [r.get('index') for r in results if isinstance(r, dict) and 'index' in r]
                print(f"Warning: Task with index {task_index} not found in results. Available indices: {available_indices}.")
                
                # 如果只有一个可用的任务，自动使用它（更友好的用户体验）
                if len(available_indices) == 1:
                    auto_index = available_indices[0]
                    print(f"Auto-using available task index {auto_index} instead of {task_index}.")
                    for result_item in results:
                        if isinstance(result_item, dict) and result_item.get("index") == auto_index:
                            task_result = result_item
                            break
                else:
                    print(f"Returning default values. Please use task_index from available indices: {available_indices}")
                    return default_return
            
            status = task_result.get("status")
            
            # 检查任务状态
            if status == "COMPLETED":
                result = task_result.get("result")
                if result is not None and isinstance(result, tuple) and len(result) == 5:
                    images, video_frames, latent, text, audio = result
                    # 如果 latent 是 None，使用占位符（避免 VAE decode 错误）
                    if latent is None:
                        print(f"Warning: Task index {task_index} has None latent, using placeholder.")
                        latent = placeholder_latent
                    else:
                        # 验证 latent 格式是否正确
                        if not isinstance(latent, dict) or "samples" not in latent:
                            print(f"Warning: Task index {task_index} has invalid latent format: {type(latent)}. Expected dict with 'samples' key. Using placeholder.")
                            latent = placeholder_latent
                    print(f"Extracted result for task index {task_index} (task_id: {task_result.get('task_id')}), latent type: {type(latent)}, has samples: {isinstance(latent, dict) and 'samples' in latent if latent else False}")
                    return (images, video_frames, latent, text, audio)
                else:
                    print(f"Warning: Task index {task_index} is marked as COMPLETED but result is not available. Returning default values.")
                    return default_return
            elif status == "ERROR":
                error = task_result.get("error", "Unknown error")
                print(f"Warning: Task index {task_index} failed with error: {error}. Returning default values.")
                return default_return
            elif status in ["PENDING", "RUNNING"]:
                print(f"Warning: Task index {task_index} is still {status}. Returning default values.")
                return default_return
            else:
                print(f"Warning: Task index {task_index} has unknown status: {status}. Returning default values.")
                return default_return
        except Exception as e:
            print(f"Error in AsyncExtractResultNode: {e}. Returning default values.")
            return default_return

