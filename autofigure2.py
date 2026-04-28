"""
Paper Method 到 SVG 图标替换完整流程 (Label 模式增强版 + Box合并 + 多Prompt支持)

支持的 API Provider：
- openrouter: OpenRouter API (https://openrouter.ai/api/v1)
- bianxie: Bianxie API (https://api.bianxie.ai/v1) - 使用 OpenAI SDK
- gemini: Google Gemini 官方 API (https://ai.google.dev/)

占位符模式 (--placeholder_mode):
- none: 无特殊样式（默认黑色边框）
- box: 传入 boxlib 坐标给 LLM
- label: 灰色填充+黑色边框+序号标签 <AF>01, <AF>02...（推荐）

SAM3 多Prompt支持 (--sam_prompt):
- 支持逗号分隔的多个text prompt
- 例如: "icon,diagram,arrow,chart"
- 对每个prompt分别检测，然后合并去重结果
- boxlib.json 会记录每个box的来源prompt

Box合并功能 (--merge_threshold):
- 对SAM3检测到的重叠box进行合并去重
- 重叠比例 = 交集面积 / 较小box面积
- 默认阈值0.9，设为0表示不合并
- 跨prompt检测结果也会自动去重

流程：
1. 输入 paper method 文本，调用 Gemini 生成学术风格图片 -> figure.png
2. SAM3 分割图片，用灰色填充+黑色边框+序号标记 -> samed.png + boxlib.json
   2.1 支持多个text prompts分别检测
   2.2 合并重叠的boxes（可选，通过 --merge_threshold 控制）
3. 裁切分割区域 + RMBG2 去背景 -> icons/icon_AF01_nobg.png, icon_AF02_nobg.png...
4. 多模态调用 Gemini 生成 SVG（占位符样式与 samed.png 一致）-> template.svg
4.5. SVG 语法验证（lxml）+ LLM 修复
4.6. LLM 优化 SVG 模板（位置和样式对齐）-> optimized_template.svg
     可通过 --optimize_iterations 参数控制迭代次数（0 表示跳过优化）
4.7. 坐标系对齐：比较 figure.png 与 SVG 尺寸，计算缩放因子
5. 根据序号匹配，将透明图标替换到 SVG 占位符中 -> final.svg

使用方法：
    # 使用 Bianxie + label 模式（默认）
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --api_key "your-key"

    # 使用 OpenRouter
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --api_key "sk-or-v1-xxx" --provider openrouter

    # 使用 box 模式（传入坐标）
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --placeholder_mode box

    # 使用多个 SAM3 prompts 检测
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --sam_prompt "icon,diagram,arrow"

    # 跳过步骤 4.6 优化（设置迭代次数为 0）
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --optimize_iterations 0

    # 设置步骤 4.6 优化迭代 3 次
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --optimize_iterations 3

    # 自定义 box 合并阈值（0.8）
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --merge_threshold 0.8

    # 禁用 box 合并
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --merge_threshold 0
"""

from __future__ import annotations

import argparse
import base64
from html import escape
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import requests
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


# ============================================================================
# Provider 配置
# ============================================================================

# 保证本地自带的 sam3 包可被导入（避免未安装时的 ModuleNotFoundError）
PROJECT_ROOT = Path(__file__).resolve().parent
SAM3_LOCAL_PATH = PROJECT_ROOT / "sam3"
if SAM3_LOCAL_PATH.exists():
    sys.path.insert(0, str(SAM3_LOCAL_PATH))

PROVIDER_CONFIGS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_image_model": "google/gemini-3-pro-image-preview",
        "default_svg_model": "google/gemini-3-pro-preview",
    },
    "bianxie": {
        "base_url": "https://api.bianxie.ai/v1",
        "default_image_model": "gemini-3-pro-image-preview",
        "default_svg_model": "gemini-3-pro-preview",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_image_model": "gemini-3-pro-image-preview",
        "default_svg_model": "gemini-2.5-pro",
    },
}

ProviderType = str
PlaceholderMode = Literal["none", "box", "label"]
GEMINI_DEFAULT_IMAGE_SIZE = "4K"

# SAM3 API config
SAM3_FAL_API_URL = "https://fal.run/fal-ai/sam-3/image"
SAM3_ROBOFLOW_API_URL = "https://serverless.roboflow.com/sam3/concept_segment"
SAM3_API_TIMEOUT = 300

# Step 1 reference image settings (overridden by CLI)
USE_REFERENCE_IMAGE = False
REFERENCE_IMAGE_PATH: Optional[str] = None


def normalize_provider(provider: Optional[str], fallback: str = "bianxie") -> str:
    provider_value = (provider or fallback).strip().lower()
    return provider_value or fallback


def resolve_llm_config(
    provider: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
    model_kind: Literal["image", "svg", "fix_svg"],
    fallback_provider: str = "bianxie",
) -> dict[str, str]:
    resolved_provider = normalize_provider(provider, fallback_provider)
    config = PROVIDER_CONFIGS.get(resolved_provider, {})
    if model_kind == "image":
        default_model_key = "default_image_model"
    elif model_kind == "svg":
        default_model_key = "default_svg_model"
    else:
        default_model_key = "default_svg_model" # fallback to svg model if fix_svg is not explicitly defined in config
    resolved_model = model or config.get(default_model_key)
    resolved_base_url = base_url or config.get("base_url")

    if not api_key:
        raise ValueError(f"必须提供{model_kind}链路的 api_key")
    if not resolved_model:
        raise ValueError(f"必须提供{model_kind}链路的 model")
    if resolved_provider != "gemini" and not resolved_base_url:
        raise ValueError(f"必须提供{model_kind}链路的 base_url")

    return {
        "provider": resolved_provider,
        "api_key": api_key,
        "base_url": resolved_base_url or "",
        "model": resolved_model,
    }


# ============================================================================
# 统一的 LLM 调用接口
# ============================================================================

def call_llm_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 160000,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    统一的文本 LLM 调用接口

    Args:
        prompt: 文本提示
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        reference_image: 参考图片（可选）
        max_tokens: 最大输出 token 数
        temperature: 温度参数

    Returns:
        LLM 响应文本
    """
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "gemini":
        # 如果提供了 base_url，说明使用第三方服务，应该用 OpenAI 兼容模式
        if base_url:
            print(f"[Gemini] 检测到自定义 base_url，使用 OpenAI 兼容模式调用: {base_url}")
            return _call_openai_compatible_text(prompt, api_key, model, base_url, "gemini", max_tokens, temperature)
        # 否则使用 Google 官方 SDK
        return _call_gemini_text(prompt, api_key, model, base_url, max_tokens, temperature)
    if normalized_provider == "openrouter":
        return _call_openrouter_text(prompt, api_key, model, base_url, max_tokens, temperature)
    if normalized_provider == "anthropic":
        return _call_anthropic_text(prompt, api_key, model, base_url, max_tokens, temperature)
    return _call_openai_compatible_text(prompt, api_key, model, base_url, normalized_provider, max_tokens, temperature)


def call_llm_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 160000,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    统一的多模态 LLM 调用接口

    Args:
        contents: 内容列表（字符串或 PIL Image）
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        max_tokens: 最大输出 token 数
        temperature: 温度参数

    Returns:
        LLM 响应文本
    """
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "gemini":
        # 如果提供了 base_url，说明使用第三方服务，应该用 OpenAI 兼容模式
        if base_url:
            print(f"[Gemini] 检测到自定义 base_url，使用 OpenAI 兼容模式调用: {base_url}")
            return _call_openai_compatible_multimodal(contents, api_key, model, base_url, "gemini", max_tokens, temperature)
        # 否则使用 Google 官方 SDK
        return _call_gemini_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    if normalized_provider == "openrouter":
        return _call_openrouter_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    if normalized_provider == "anthropic":
        return _call_anthropic_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    return _call_openai_compatible_multimodal(contents, api_key, model, base_url, normalized_provider, max_tokens, temperature)


def call_llm_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """
    统一的图像生成 LLM 调用接口

    Args:
        prompt: 文本提示
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商

    Returns:
        生成的 PIL Image，失败返回 None
    """
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "gemini":
        # 如果提供了 base_url，说明使用第三方服务，应该用 OpenAI 兼容模式
        if base_url:
            print(f"[Gemini] 检测到自定义 base_url，使用 OpenAI 兼容模式调用: {base_url}")
            return _call_openai_compatible_image_generation(prompt, api_key, model, base_url, "gemini", reference_image)
        # 否则使用 Google 官方 SDK
        return _call_gemini_image_generation(
            prompt=prompt,
            api_key=api_key,
            model=model,
            base_url=base_url,
            reference_image=reference_image,
            image_size=GEMINI_DEFAULT_IMAGE_SIZE,
        )
    if normalized_provider == "openrouter":
        return _call_openrouter_image_generation(prompt, api_key, model, base_url, reference_image)
    if normalized_provider == "anthropic":
        print("[Anthropic] 暂不支持原生生图接口，尝试回退到 OpenAI 兼容模式调用")
        return _call_openai_compatible_image_generation(prompt, api_key, model, base_url, normalized_provider, reference_image)
    return _call_openai_compatible_image_generation(prompt, api_key, model, base_url, normalized_provider, reference_image)


# ============================================================================
# OpenAI-compatible Provider 实现
# ============================================================================

def _call_openai_compatible_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider_label: str,
    max_tokens: int = 160000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 OpenAI SDK 调用兼容接口"""
    try:
        from openai import OpenAI
        
        # 确保 base_url 以版本路径结尾（如 /v1, /v3），但不重复追加
        normalized_base_url = base_url.rstrip('/')
        if not re.search(r'/v\d+$', normalized_base_url):
            normalized_base_url = f"{normalized_base_url}/v1"
        
        client = OpenAI(
            api_key=api_key,
            base_url=normalized_base_url,
            timeout=600.0,    # 600 seconds total timeout
            max_retries=0,     # disable auto retries to avoid repeated charges
        )
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[{provider_label}] API 调用失败: {e}")
        raise


def _is_deepseek_non_vision(base_url: str, model: str) -> bool:
    """检测是否为 DeepSeek 非视觉模型（不支持 image_url）"""
    if "deepseek" not in base_url.lower():
        return False
    # DeepSeek 视觉模型名称中通常包含 "vl"，如 deepseek-vl2, deepseek-vl2-plus
    model_lower = model.lower()
    if "vl" in model_lower:
        return False
    return True


def _call_openai_compatible_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    provider_label: str,
    max_tokens: int = 160000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 OpenAI SDK 调用兼容多模态接口"""
    try:
        from openai import OpenAI
        
        # 确保 base_url 以版本路径结尾（如 /v1, /v3），但不重复追加
        normalized_base_url = base_url.rstrip('/')
        if not re.search(r'/v\d+$', normalized_base_url):
            normalized_base_url = f"{normalized_base_url}/v1"
        
        # DeepSeek 非视觉模型不支持 image_url，需要剥离图片内容
        strip_images = _is_deepseek_non_vision(base_url, model)
        if strip_images:
            print(f"[{provider_label}] 模型 {model} 不支持图片输入，将仅发送文本内容")
        
        client = OpenAI(
            api_key=api_key,
            base_url=normalized_base_url,
            timeout=600.0,    # 600 seconds total timeout
            max_retries=0,     # disable auto retries to avoid repeated charges
        )
        
        message_content: List[Dict[str, Any]] = []
        for part in contents:
            if isinstance(part, str):
                message_content.append({"type": "text", "text": part})
            elif isinstance(part, Image.Image):
                if strip_images:
                    print(f"[{provider_label}] 跳过图片内容（模型不支持视觉输入）")
                    continue
                buf = io.BytesIO()
                part.save(buf, format='PNG')
                image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[{provider_label}] 多模态 API 调用失败: {e}")
        raise


def _call_openai_compatible_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider_label: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 OpenAI SDK 调用兼容图像生成接口"""
    try:
        from openai import OpenAI
        import re
        
        # 确保 base_url 以版本路径结尾（如 /v1, /v3），但不重复追加
        normalized_base_url = base_url.rstrip('/')
        if not re.search(r'/v\d+$', normalized_base_url):
            normalized_base_url = f"{normalized_base_url}/v1"
            
        import httpx
        client = OpenAI(
            api_key=api_key,
            base_url=normalized_base_url,
            timeout=httpx.Timeout(connect=30.0, read=None, write=60.0, pool=10.0),  # read=None: no read timeout for slow image generation
            max_retries=0,     # disable auto retries to avoid repeated charges
        )

        if "gemini" in model.lower() and "preview" in model.lower():
            print(f"[{provider_label}] 检测到 {model} 模型，使用 Chat 模式调用图像生成 API")
            
            content = []
            if reference_image is not None:
                buf = io.BytesIO()
                reference_image.save(buf, format='PNG')
                image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })
            
            content.append({"type": "text", "text": prompt})
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}]
            )
            
            if response.choices and response.choices[0].message.content:
                content_text = response.choices[0].message.content
                match = re.search(r'!\[.*?\]\(data:image/(png|jpeg);base64,(.*?)\)', content_text)
                if match:
                    base64_data = match.group(2)
                    img_data = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(img_data)).convert("RGB")
                else:
                    print(f"[{provider_label}] 未找到合法的 base64 图片数据")
                    return None
            return None
        
        # 注意：reference_image 在标准 OpenAI images API 中不支持
        if reference_image is not None:
            print(f"[{provider_label}] 警告：标准图像生成 API 不支持参考图片，将忽略 reference_image")
        
        print(f"[{provider_label}] 调用图像生成 API: {normalized_base_url}/images/generations")
        
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            response_format="b64_json"
        )
        
        # 处理响应
        if hasattr(response, 'data') and response.data:
            b64_data = response.data[0].b64_json
            img_data = base64.b64decode(b64_data)
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            print(f"[{provider_label}] 生图失败，未返回图像数据")
            return None
    except Exception as e:
        print(f"[{provider_label}] 图像生成 API 调用失败: {e}")
        raise


# ============================================================================
# OpenRouter Provider 实现 (使用 requests)
# ============================================================================

def _get_openrouter_headers(api_key: str) -> dict:
    """获取 OpenRouter 请求头"""
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'https://localhost',
        'X-Title': 'MethodToSVG'
    }


def _get_openrouter_api_url(base_url: str) -> str:
    """获取 OpenRouter API URL"""
    if not base_url.endswith('/chat/completions'):
        if base_url.endswith('/'):
            return base_url + 'chat/completions'
        else:
            return base_url + '/chat/completions'
    return base_url


def _call_openrouter_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 160000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 requests 调用 OpenRouter 文本接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API 错误: {error_msg}')

    choices = result.get('choices', [])
    if not choices:
        return None

    return choices[0].get('message', {}).get('content', '')


def _call_openrouter_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 160000,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 requests 调用 OpenRouter 多模态接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    message_content: List[Dict[str, Any]] = []
    for part in contents:
        if isinstance(part, str):
            message_content.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            buf = io.BytesIO()
            part.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': message_content}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API 错误: {error_msg}')

    choices = result.get('choices', [])
    if not choices:
        return None

    return choices[0].get('message', {}).get('content', '')


def _call_openrouter_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 requests 调用 OpenRouter 图像生成接口"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    if reference_image is None:
        messages = [{'role': 'user', 'content': prompt}]
    else:
        buf = io.BytesIO()
        reference_image.save(buf, format='PNG')
        image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        messages = [{'role': 'user', 'content': message_content}]

    payload = {
        'model': model,
        'messages': messages,
        'modalities': ['image', 'text'],
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API 错误: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API 错误: {error_msg}')

    # OpenRouter 返回图片在 message.images[] 数组中
    choices = result.get('choices', [])
    if not choices:
        return None

    message = choices[0].get('message', {})
    images = message.get('images', [])

    if images and len(images) > 0:
        first_image = images[0]

        if isinstance(first_image, dict):
            image_url_obj = first_image.get('image_url', {})
            if isinstance(image_url_obj, dict):
                image_url = image_url_obj.get('url', '')
            else:
                image_url = str(image_url_obj)
        else:
            image_url = str(first_image)

        if image_url.startswith('data:image/'):
            pattern = r'data:image/(png|jpeg|jpg|webp);base64,(.+)'
            match = re.match(pattern, image_url)
            if match:
                image_base64 = match.group(2)
                image_data = base64.b64decode(image_base64)
                return Image.open(io.BytesIO(image_data))

    return None


# ============================================================================
# Anthropic Provider 实现 (使用 requests 调用原生 Messages API)
# ============================================================================

def _call_anthropic_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 Anthropic HTTP API 调用纯文本接口"""
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        endpoint = base_url.rstrip('/')
        if not endpoint.endswith('v1/messages'):
            endpoint = f"{endpoint}/v1/messages"
            
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "content" in result and len(result["content"]) > 0:
            for block in result["content"]:
                if isinstance(block, dict) and "text" in block:
                    return block["text"]
        return None
    except Exception as e:
        print(f"[Anthropic] 文本 API 调用失败: {e}")
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
            print(f"[Anthropic] Error Response: {e.response.text}")
        raise


def _call_anthropic_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> Optional[str]:
    """使用 Anthropic HTTP API 调用多模态接口"""
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        message_content = []
        for part in contents:
            if isinstance(part, str):
                message_content.append({
                    "type": "text",
                    "text": part
                })
            elif isinstance(part, Image.Image):
                buf = io.BytesIO()
                part.save(buf, format='PNG')
                image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64
                    }
                })

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": message_content}
            ]
        }
        
        endpoint = base_url.rstrip('/')
        if not endpoint.endswith('v1/messages'):
            endpoint = f"{endpoint}/v1/messages"
            
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "content" in result and len(result["content"]) > 0:
            for block in result["content"]:
                if isinstance(block, dict) and "text" in block:
                    return block["text"]
        return None
    except Exception as e:
        print(f"[Anthropic] 多模态 API 调用失败: {e}")
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
            print(f"[Anthropic] Error Response: {e.response.text}")
        raise


# ============================================================================
# Gemini Provider 实现 (Google 官方 SDK)
# ============================================================================

def _get_gemini_client(api_key: str, base_url: Optional[str] = None):
    """获取 Gemini 客户端（延迟导入，避免非 Gemini 场景强依赖）"""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "未安装 google-genai，请执行: pip install google-genai"
        ) from e
    if base_url:
        # 使用自定义 API 端点（适用于第三方服务）
        http_options = types.HttpOptions(baseUrl=base_url)
        return genai.Client(api_key=api_key, http_options=http_options)
    return genai.Client(api_key=api_key)


def _build_gemini_text_config(max_tokens: int, temperature: float):
    """构建 Gemini 文本生成配置"""
    from google.genai import types

    return types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )


def _extract_gemini_text(response: Any) -> Optional[str]:
    """从 Gemini 响应中提取文本"""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    parts = getattr(response, "parts", None) or []
    extracted: list[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if isinstance(part_text, str) and part_text.strip():
            extracted.append(part_text)
    if extracted:
        return "\n".join(extracted)

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                extracted.append(part_text)
    if extracted:
        return "\n".join(extracted)

    return None


def _extract_gemini_image(response: Any) -> Optional[Image.Image]:
    """从 Gemini 响应中提取图片（优先使用 part.as_image()）"""
    parts = getattr(response, "parts", None) or []
    for part in parts:
        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            image = as_image()
            if image is not None:
                return image

        inline_data = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline_data is None:
            continue
        data = getattr(inline_data, "data", None)
        if isinstance(data, bytes) and data:
            return Image.open(io.BytesIO(data))
        if isinstance(data, str) and data:
            return Image.open(io.BytesIO(base64.b64decode(data)))

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            as_image = getattr(part, "as_image", None)
            if callable(as_image):
                image = as_image()
                if image is not None:
                    return image
    return None


def _call_gemini_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    max_tokens: int = 160000,
    temperature: float = 0.7,
) -> Optional[str]:
    """调用 Gemini 文本接口"""
    try:
        client = _get_gemini_client(api_key, base_url)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_build_gemini_text_config(max_tokens=max_tokens, temperature=temperature),
        )
        return _extract_gemini_text(response)
    except Exception as e:
        print(f"[Gemini] 文本 API 调用失败: {e}")
        raise


def _call_gemini_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    max_tokens: int = 160000,
    temperature: float = 0.7,
) -> Optional[str]:
    """调用 Gemini 多模态接口"""
    try:
        client = _get_gemini_client(api_key, base_url)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=_build_gemini_text_config(max_tokens=max_tokens, temperature=temperature),
        )
        return _extract_gemini_text(response)
    except Exception as e:
        print(f"[Gemini] 多模态 API 调用失败: {e}")
        raise


def _call_gemini_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    reference_image: Optional[Image.Image] = None,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
) -> Optional[Image.Image]:
    """调用 Gemini 生图接口，默认 image_size=4K"""
    try:
        from google.genai import types

        client = _get_gemini_client(api_key, base_url)
        config = types.GenerateContentConfig(
            image_config=types.ImageConfig(image_size=image_size),
        )

        if reference_image is None:
            contents: list[Any] = [prompt]
        else:
            # 参考图放在前面，提示语在后，遵循 Gemini 多模态输入习惯
            contents = [reference_image, prompt]

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return _extract_gemini_image(response)
    except Exception as e:
        print(f"[Gemini] 图像生成 API 调用失败: {e}")
        raise


# ============================================================================
# 步骤一：调用 LLM 生成图片
# ============================================================================

def generate_figure_from_method(
    method_text: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    use_reference_image: Optional[bool] = None,
    reference_image_path: Optional[str] = None,
) -> str:
    """
    使用 LLM 生成学术风格图片

    Args:
        method_text: Paper method 文本内容
        output_path: 输出图片路径
        api_key: API Key
        model: 生图模型名称
        base_url: API base URL
        provider: API 提供商
        use_reference_image: 是否使用参考图片（None 则使用全局设置）
        reference_image_path: 参考图片路径（None 则使用全局设置）

    Returns:
        生成的图片路径
    """
    # 检查全局 output 目录中是否存在 figure.png
    global_figure_path = Path("output/figure.png")
    if not os.path.exists(output_path) and global_figure_path.exists():
        import shutil
        print(f"[{provider}] 发现全局目录下的已存在图片: {global_figure_path}，复制到当前任务目录并跳过 API 生成步骤")
        shutil.copy(str(global_figure_path), output_path)
        return output_path

    if os.path.exists(output_path):
        print(f"[{provider}] 发现已存在的图片: {output_path}，跳过 API 生成步骤以节省额度")
        return output_path
    print("=" * 60)
    print("步骤一：使用 LLM 生成学术风格图片")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")

    if use_reference_image is None:
        use_reference_image = USE_REFERENCE_IMAGE
    if reference_image_path is None:
        reference_image_path = REFERENCE_IMAGE_PATH
    if reference_image_path:
        use_reference_image = True

    reference_image = None
    if use_reference_image:
        if not reference_image_path:
            raise ValueError("启用参考图模式但未提供 reference_image_path")
        reference_image = Image.open(reference_image_path)
        print(f"参考图片: {reference_image_path}")

    if use_reference_image:
        prompt = f"""Generate a figure to visualize the method described below.

You should closely imitate the visual (artistic) style of the reference figure I provide, focusing only on aesthetic aspects, NOT on layout or structure.

Specifically, match:
- overall visual tone and mood
- illustration abstraction level
- line style
- color usage
- shading style
- icon and shape style
- arrow and connector aesthetics
- typography feel

The content structure, number of components, and layout may differ freely.
Only the visual style should be consistent.

The goal is that the figure looks like it was drawn by the same illustrator using the same visual design language as the reference figure.

Below is the method section of the paper:
\"\"\"
{method_text}
\"\"\""""
    else:
        prompt = f"""Generate a professional academic journal style figure for the paper below so as to visualize the method it proposes, below is the method section of this paper:

{method_text}

The figure should be engaging and using academic journal style with cute characters."""

    print(f"发送请求到: {base_url}")

    img = call_llm_image_generation(
        prompt=prompt,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        reference_image=reference_image,
    )

    if img is None:
        raise Exception('API 响应中没有找到图片')

    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为 PNG 保存（Gemini 返回的图片对象 save() 可能不接受 format 参数）
    try:
        img.save(str(output_path), format='PNG')
    except TypeError:
        img.save(str(output_path))
        # 某些 SDK 对象会按自身默认编码写盘（如 JPEG），这里强制转存为真实 PNG
        with Image.open(str(output_path)) as normalized:
            normalized.save(str(output_path), format='PNG')
    print(f"图片已保存: {output_path}")
    return str(output_path)


# ============================================================================
# 步骤二：SAM3 分割 + Box合并 + 灰色填充+黑色边框+序号标记
# ============================================================================

def get_label_font(box_width: int, box_height: int) -> ImageFont.FreeTypeFont:
    """
    根据 box 尺寸动态计算合适的字体大小

    Args:
        box_width: 矩形宽度
        box_height: 矩形高度

    Returns:
        PIL ImageFont 对象
    """
    # 字体大小为 box 短边的 1/4，最小 12，最大 48
    min_dim = min(box_width, box_height)
    font_size = max(12, min(48, min_dim // 4))

    # 尝试加载字体
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:/Windows/Fonts/arial.ttf",  # Windows
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            continue

    # 回退到默认字体
    try:
        return ImageFont.load_default()
    except:
        return None


# ============================================================================
# Box 合并辅助函数
# ============================================================================

def calculate_overlap_ratio(box1: dict, box2: dict) -> float:
    """
    计算两个box的重叠比例
    
    Args:
        box1: 第一个box，包含 x1, y1, x2, y2
        box2: 第二个box，包含 x1, y1, x2, y2

    Returns:
        重叠比例 = 交集面积 / 较小box面积 (如果面积相差悬殊则返回0，避免大框吞噬小框)
    """
    # 计算交集区域
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    # 无交集
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算各自面积
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    if area1 == 0 or area2 == 0:
        return 0.0

    # 如果面积相差过大（例如大框面积是小框的 4 倍以上），说明是包含/父子关系，不应合并
    if max(area1, area2) / min(area1, area2) > 4.0:
        return 0.0

    # 返回交集占较小box的比例
    return intersection / min(area1, area2)


def merge_two_boxes(box1: dict, box2: dict) -> dict:
    """
    合并两个box为最小包围矩形

    Args:
        box1: 第一个box
        box2: 第二个box

    Returns:
        合并后的box（最小包围矩形）
    """
    merged = {
        "x1": min(box1["x1"], box2["x1"]),
        "y1": min(box1["y1"], box2["y1"]),
        "x2": max(box1["x2"], box2["x2"]),
        "y2": max(box1["y2"], box2["y2"]),
        "score": max(box1.get("score", 0), box2.get("score", 0)),  # 保留较高置信度
    }
    # 合并 prompt 字段（如果存在）
    prompt1 = box1.get("prompt", "")
    prompt2 = box2.get("prompt", "")
    if prompt1 and prompt2:
        if prompt1 == prompt2:
            merged["prompt"] = prompt1
        else:
            # 合并不同的 prompts，保留置信度更高的那个
            if box1.get("score", 0) >= box2.get("score", 0):
                merged["prompt"] = prompt1
            else:
                merged["prompt"] = prompt2
    elif prompt1:
        merged["prompt"] = prompt1
    elif prompt2:
        merged["prompt"] = prompt2
    return merged


def merge_overlapping_boxes(
    boxes: list,
    overlap_threshold: float = 0.3,
    image_size: Optional[tuple[int, int]] = None,
    max_merged_area_ratio: float = 0.1,
) -> list:
    """
    迭代合并重叠的boxes

    Args:
        boxes: box列表，每个box包含 x1, y1, x2, y2, score
        overlap_threshold: 重叠阈值，超过此值则合并（默认0.9）

    Returns:
        合并后的box列表，重新编号
    """
    if overlap_threshold <= 0 or len(boxes) <= 1:
        return boxes

    # 复制列表避免修改原数据
    working_boxes = [box.copy() for box in boxes]
    img_area = None
    if image_size and len(image_size) >= 2:
        img_w, img_h = image_size
        if img_w > 0 and img_h > 0:
            img_area = img_w * img_h

    merged = True
    iteration = 0
    while merged:
        merged = False
        iteration += 1
        n = len(working_boxes)

        for i in range(n):
            if merged:
                break
            for j in range(i + 1, n):
                ratio = calculate_overlap_ratio(working_boxes[i], working_boxes[j])
                if ratio >= overlap_threshold:
                    new_box = merge_two_boxes(working_boxes[i], working_boxes[j])
                    if img_area and max_merged_area_ratio > 0:
                        merged_area = (new_box["x2"] - new_box["x1"]) * (new_box["y2"] - new_box["y1"])
                        if merged_area > img_area * max_merged_area_ratio:
                            print(
                                f"    迭代 {iteration}: 取消合并 box {i} 和 box {j} "
                                f"(合并后面积占比: {merged_area / img_area:.1%} 超过 {max_merged_area_ratio:.1%})"
                            )
                            continue
                    working_boxes = [
                        working_boxes[k] for k in range(n) if k != i and k != j
                    ]
                    working_boxes.append(new_box)
                    merged = True
                    print(f"    迭代 {iteration}: 合并 box {i} 和 box {j} (重叠比例: {ratio:.2f})")
                    break

    # 重新编号
    result = []
    for idx, box in enumerate(working_boxes):
        result_box = {
            "id": idx,
            "label": f"<AF>{idx + 1:02d}",
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
            "score": box.get("score", 0),
        }
        # 保留 prompt 字段（如果存在）
        if "prompt" in box:
            result_box["prompt"] = box["prompt"]
        result.append(result_box)

    return result


def _get_fal_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("FAL_KEY")
    if not key:
        raise ValueError("SAM3 fal.ai API key missing: set --sam_api_key or FAL_KEY environment variable")
    return key


def _get_roboflow_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY")
    if not key:
        raise ValueError(
            "SAM3 Roboflow API key missing: set --sam_api_key or ROBOFLOW_API_KEY/API_KEY environment variable"
        )
    return key


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _cxcywh_norm_to_xyxy(box: list | tuple, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if not box or len(box) < 4:
        return None
    try:
        cx, cy, bw, bh = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None

    cx *= width
    cy *= height
    bw *= width
    bh *= height

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _polygon_to_bbox(points: list, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    xs: list[float] = []
    ys: list[float] = []

    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_sam3_api_detections(response_json: dict, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []

    metadata = response_json.get("metadata") if isinstance(response_json, dict) else None
    if isinstance(metadata, list) and metadata:
        for item in metadata:
            if not isinstance(item, dict):
                continue
            box = item.get("box")
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = item.get("score")
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )
        return detections

    boxes = response_json.get("boxes") if isinstance(response_json, dict) else None
    scores = response_json.get("scores") if isinstance(response_json, dict) else None
    if isinstance(boxes, list) and boxes:
        scores_list = scores if isinstance(scores, list) else []
        for idx, box in enumerate(boxes):
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = scores_list[idx] if idx < len(scores_list) else None
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )

    return detections


def _extract_roboflow_detections(response_json: dict, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []

    prompt_results = response_json.get("prompt_results") if isinstance(response_json, dict) else None
    if not isinstance(prompt_results, list):
        return detections

    for prompt_result in prompt_results:
        if not isinstance(prompt_result, dict):
            continue
        predictions = prompt_result.get("predictions", [])
        if not isinstance(predictions, list):
            continue
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            confidence = prediction.get("confidence")
            masks = prediction.get("masks", [])
            if not isinstance(masks, list):
                continue
            for mask in masks:
                points = []
                if isinstance(mask, list) and mask:
                    if isinstance(mask[0], (list, tuple)) and len(mask[0]) >= 2 and isinstance(
                        mask[0][0], (int, float)
                    ):
                        points = mask
                    elif isinstance(mask[0], (list, tuple)):
                        for sub in mask:
                            if isinstance(sub, (list, tuple)) and len(sub) >= 2 and isinstance(
                                sub[0], (int, float)
                            ):
                                points.append(sub)
                            elif isinstance(sub, (list, tuple)) and sub and isinstance(
                                sub[0], (list, tuple)
                            ):
                                for pt in sub:
                                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                        points.append(pt)
                if not points:
                    continue
                xyxy = _polygon_to_bbox(points, width, height)
                if not xyxy:
                    continue
                detections.append(
                    {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "score": confidence,
                    }
                )

    return detections


def _call_sam3_api(
    image_data_uri: str,
    prompt: str,
    api_key: str,
    max_masks: int,
) -> dict:
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "image_url": image_data_uri,
        "prompt": prompt,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": max_masks,
        "include_scores": True,
        "include_boxes": True,
    }
    response = requests.post(SAM3_FAL_API_URL, headers=headers, json=payload, timeout=SAM3_API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"SAM3 API 错误: {response.status_code} - {response.text[:500]}")
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"SAM3 API 错误: {result.get('error')}")
    return result


def _call_sam3_roboflow_api(
    image_base64: str,
    prompt: str,
    api_key: str,
    min_score: float,
) -> dict:
    payload = {
        "image": {"type": "base64", "value": image_base64},
        "prompts": [{"type": "text", "text": prompt}],
        "format": "polygon",
        "output_prob_thresh": min_score,
    }
    url = f"{SAM3_ROBOFLOW_API_URL}?api_key={api_key}"
    response = requests.post(url, json=payload, timeout=SAM3_API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"SAM3 Roboflow API 错误: {response.status_code} - {response.text[:500]}")
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"SAM3 Roboflow API 错误: {result.get('error')}")
    return result
def segment_with_sam3(
    image_path: str,
    output_dir: str,
    text_prompts: str = "icon",
    min_score: float = 0.2,
    merge_threshold: float = 0.3,
    max_box_area_ratio: float = 0.1,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
    enable_enhanced_detection: bool = False,
) -> tuple[str, str, list]:
    """
    使用 SAM3 分割图片，用灰色填充+黑色边框+序号标记，生成 boxlib.json

    占位符样式：
    - 灰色填充 (#808080)
    - 黑色边框 (width=3)
    - 白色居中序号标签 (<AF>01, <AF>02, ...)

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        text_prompts: SAM3 文本提示，支持逗号分隔的多个prompt（如 "icon,diagram,arrow"）
        min_score: 最低置信度阈值
        merge_threshold: Box合并阈值，重叠比例超过此值则合并（0表示不合并，默认0.9）

    Returns:
        (samed_path, boxlib_path, valid_boxes)
    """
    print("\n" + "=" * 60)
    print("步骤二：SAM3 分割 + 灰色填充+黑色边框+序号标记")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    original_size = image.size
    print(f"原图尺寸: {original_size[0]} x {original_size[1]}")

    # 解析多个 prompts（支持逗号分隔）
    prompt_list = [p.strip() for p in text_prompts.split(",") if p.strip()]
    print(f"使用的 prompts: {prompt_list}")

    # 对每个 prompt 分别检测并收集结果
    all_detected_boxes = []
    total_detected = 0

    backend = sam_backend
    if backend == "api":
        backend = "fal"

    if backend == "local":
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import sam3

        sam3_dir = Path(sam3.__path__[0]) if hasattr(sam3, '__path__') else Path(sam3.__file__).parent
        bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            bpe_path = None
            print("警告: 未找到 bpe 文件，使用默认路径")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        local_sam3_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sam3", "sam3.pt")
        if os.path.exists(local_sam3_ckpt):
            print(f"加载本地 SAM3 权重: {local_sam3_ckpt}")
            model = build_sam3_image_model(device=device, bpe_path=str(bpe_path) if bpe_path else None, checkpoint_path=local_sam3_ckpt, load_from_HF=False)
        else:
            model = build_sam3_image_model(device=device, bpe_path=str(bpe_path) if bpe_path else None)
            
        processor = Sam3Processor(model, device=device, confidence_threshold=min_score)
        inference_state = processor.set_image(image.convert("RGB") if image.mode != "RGB" else image)

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

            boxes = output["boxes"]
            scores = output["scores"]

            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            prompt_count = 0
            img_area = original_size[0] * original_size[1]
            for box, score in zip(boxes, scores):
                if score >= min_score:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    box_area = (x2 - x1) * (y2 - y1)
                    if img_area > 0 and (box_area / img_area) > 0.7:
                        print(f"    跳过: 框过大 ({box_area/img_area:.1%} 超过70%图像面积)")
                        continue
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": float(score),
                        "prompt": prompt  # 记录来源 prompt
                    })
                    prompt_count += 1
                    print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score:.3f}")
                else:
                    print(f"    跳过: score={score:.3f} < {min_score}")

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif backend == "fal":
        api_key = _get_fal_api_key(sam_api_key)
        max_masks = max(1, min(32, int(sam_max_masks)))
        image_data_uri = _image_to_data_uri(image)
        print(f"SAM3 fal.ai API 模式: max_masks={max_masks}")

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            response_json = _call_sam3_api(
                image_data_uri=image_data_uri,
                prompt=prompt,
                api_key=api_key,
                max_masks=max_masks,
            )
            detections = _extract_sam3_api_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    # 过滤掉过大的框
                    box_area = (x2 - x1) * (y2 - y1)
                    img_area = original_size[0] * original_size[1]
                    if img_area > 0 and (box_area / img_area) > 0.7:
                        continue
                        
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score_val,
                        "prompt": prompt  # 记录来源 prompt
                    })
                    prompt_count += 1
                    print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")
                else:
                    print(f"    跳过: score={score_val:.3f} < {min_score}")

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count
    elif backend == "roboflow":
        api_key = _get_roboflow_api_key(sam_api_key)
        image_base64 = _image_to_base64(image)
        print("SAM3 Roboflow API 模式: format=polygon")

        for prompt in prompt_list:
            print(f"\n  正在检测: '{prompt}'")
            response_json = _call_sam3_roboflow_api(
                image_base64=image_base64,
                prompt=prompt,
                api_key=api_key,
                min_score=min_score,
            )
            detections = _extract_roboflow_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    # 过滤掉过大的框
                    box_area = (x2 - x1) * (y2 - y1)
                    img_area = original_size[0] * original_size[1]
                    if img_area > 0 and (box_area / img_area) > 0.7:
                        continue
                        
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score_val,
                        "prompt": prompt
                    })
                    prompt_count += 1
                    print(f"    对象 {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")
                else:
                    print(f"    跳过: score={score_val:.3f} < {min_score}")

            print(f"  '{prompt}' 检测到 {prompt_count} 个有效对象")
            total_detected += prompt_count
    else:
        raise ValueError(f"未知 SAM3 后端: {sam_backend}")

    print(f"\n总计检测: {total_detected} 个对象 (来自 {len(prompt_list)} 个 prompts)")

    # 为所有检测到的 boxes 分配临时 id 和 label（用于合并）
    valid_boxes = []
    for i, box_data in enumerate(all_detected_boxes):
        valid_boxes.append({
            "id": i,
            "label": f"<AF>{i + 1:02d}",
            "x1": box_data["x1"],
            "y1": box_data["y1"],
            "x2": box_data["x2"],
            "y2": box_data["y2"],
            "score": box_data["score"],
            "prompt": box_data["prompt"]
        })

    # === 新增：合并重叠的boxes ===
    if merge_threshold > 0 and len(valid_boxes) > 1:
        max_area_text = f"{max_box_area_ratio:.1%}" if max_box_area_ratio > 0 else "不限制"
        print(f"\n  合并重叠的boxes (阈值: {merge_threshold}, 最大面积: {max_area_text})...")
        original_count = len(valid_boxes)
        valid_boxes = merge_overlapping_boxes(
            valid_boxes,
            merge_threshold,
            image_size=original_size,
            max_merged_area_ratio=max_box_area_ratio,
        )
        merged_count = original_count - len(valid_boxes)
        if merged_count > 0:
            print(f"  合并完成: {original_count} -> {len(valid_boxes)} (合并了 {merged_count} 个)")
            # 打印合并后的box信息
            print(f"\n  合并后的boxes:")
            for box_info in valid_boxes:
                print(f"    {box_info['label']}: ({box_info['x1']}, {box_info['y1']}, {box_info['x2']}, {box_info['y2']})")
        else:
            print(f"  无需合并，所有boxes重叠比例均低于阈值")
    
    # === 新增：增强检测补充遗漏元素 ===
    if enable_enhanced_detection:
        try:
            from enhanced_detection import enhance_sam_detection
            print(f"\n  启用增强检测补充遗漏元素...")
            valid_boxes = enhance_sam_detection(image, valid_boxes)
            # 重新分配ID和label
            for i, box in enumerate(valid_boxes):
                box["id"] = i
                box["label"] = f"<AF>{i + 1:02d}"
                if "prompt" not in box:
                    box["prompt"] = "enhanced"
        except ImportError:
            print(f"  警告: enhanced_detection模块未找到，跳过增强检测")
        except Exception as e:
            print(f"  警告: 增强检测失败: {e}")

    final_max_box_area_ratio = max_box_area_ratio
    img_area = original_size[0] * original_size[1]
    if img_area > 0 and valid_boxes and final_max_box_area_ratio > 0:
        filtered_boxes = []
        removed_large_boxes = []
        for box in valid_boxes:
            box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
            if (box_area / img_area) > final_max_box_area_ratio:
                removed_large_boxes.append((box, box_area / img_area))
            else:
                filtered_boxes.append(box)

        if removed_large_boxes:
            print(f"\n  最终box面积过滤 (上限: {final_max_box_area_ratio:.1%})...")
            for box, area_ratio in removed_large_boxes:
                print(
                    f"    跳过最终大框: {box['label']} ({box['x1']}, {box['y1']}, {box['x2']}, {box['y2']}), "
                    f"score={box.get('score', 0):.3f}, prompt={box.get('prompt', '')}, 面积占比={area_ratio:.1%}"
                )
            valid_boxes = filtered_boxes
            for i, box in enumerate(valid_boxes):
                box["id"] = i
                box["label"] = f"<AF>{i + 1:02d}"

    # 使用合并后的 valid_boxes 创建标记图片
    print(f"\n  绘制 samed.png (使用 {len(valid_boxes)} 个boxes)...")
    samed_image = image.copy()
    draw = ImageDraw.Draw(samed_image)

    for box_info in valid_boxes:
        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]
        label = box_info["label"]

        # 灰色填充 + 黑色边框
        draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)

        # 计算中心点
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 获取合适大小的字体
        box_width = x2 - x1
        box_height = y2 - y1
        font = get_label_font(box_width, box_height)

        # 绘制白色居中序号标签
        if font:
            # 使用 anchor="mm" 居中绘制（如果支持）
            try:
                draw.text((cx, cy), label, fill="white", anchor="mm", font=font)
            except TypeError:
                # 旧版本 PIL 不支持 anchor，手动计算位置
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = cx - text_width // 2
                text_y = cy - text_height // 2
                draw.text((text_x, text_y), label, fill="white", font=font)
        else:
            # 无字体时使用默认
            draw.text((cx, cy), label, fill="white")

    samed_path = output_dir / "samed.png"
    samed_image.save(str(samed_path))
    print(f"标记图片已保存: {samed_path}")

    boxlib_data = {
        "image_size": {"width": original_size[0], "height": original_size[1]},
        "prompts_used": prompt_list,
        "boxes": valid_boxes
    }

    boxlib_path = output_dir / "boxlib.json"
    with open(boxlib_path, 'w', encoding='utf-8') as f:
        json.dump(boxlib_data, f, indent=2, ensure_ascii=False)
    print(f"Box 信息已保存: {boxlib_path}")

    return str(samed_path), str(boxlib_path), valid_boxes


# ============================================================================
# 步骤三：裁切 + RMBG2 去背景
# ============================================================================

class BriaRMBG2Remover:
    """使用 BRIA-RMBG 2.0 模型进行高质量背景抠图"""

    def __init__(self, model_path: Optional[str] = None, output_dir: str = "icons"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir

        if not model_path:
            local_rmbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "RMBG-2.0")
            if os.path.exists(local_rmbg):
                model_path = local_rmbg

        if model_path and os.path.exists(model_path):
            print(f"加载本地 RMBG 权重: {model_path}")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                str(model_path), trust_remote_code=True,
            ).eval().to(self.device)
        else:
            print("从 HuggingFace 加载 RMBG-2.0 模型...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0", trust_remote_code=True,
            ).eval().to(device)

        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def remove_background(self, image: Image.Image, output_name: str) -> str:
        image_rgb = image.convert("RGB")
        input_tensor = self.transform_image(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_rgb.size)

        out = image_rgb.copy()
        out.putalpha(mask)

        out_path = self.output_dir / f"{output_name}_nobg.png"
        out.save(out_path)
        return str(out_path)


def crop_and_remove_background(
    image_path: str,
    boxlib_path: str,
    output_dir: str,
    rmbg_model_path: Optional[str] = None,
) -> list[dict]:
    """
    根据 boxlib.json 裁切图片并使用 RMBG2 去背景

    文件命名使用 label: icon_AF01.png, icon_AF01_nobg.png
    """
    print("\n" + "=" * 60)
    print("步骤三：裁切 + RMBG2 去背景")
    print("=" * 60)

    output_dir = Path(output_dir)
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    with open(boxlib_path, 'r', encoding='utf-8') as f:
        boxlib_data = json.load(f)

    boxes = boxlib_data["boxes"]

    if len(boxes) == 0:
        print("警告: 没有检测到有效的 box")
        return []

    remover = BriaRMBG2Remover(model_path=rmbg_model_path, output_dir=icons_dir)

    icon_infos = []
    for box_info in boxes:
        box_id = box_info["id"]
        label = box_info.get("label", f"<AF>{box_id + 1:02d}")
        # 将 <AF>01 转换为 AF01 用于文件名
        label_clean = label.replace("<", "").replace(">", "")

        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]

        cropped = image.crop((x1, y1, x2, y2))
        crop_path = icons_dir / f"icon_{label_clean}.png"
        cropped.save(crop_path)

        nobg_path = remover.remove_background(cropped, f"icon_{label_clean}")

        icon_infos.append({
            "id": box_id,
            "label": label,
            "label_clean": label_clean,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "crop_path": str(crop_path),
            "nobg_path": nobg_path,
        })

        print(f"  {label}: 裁切并去背景完成 -> {nobg_path}")

    del remover
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return icon_infos


# ============================================================================
# 步骤四：多模态调用生成 SVG
# ============================================================================

def generate_svg_template(
    figure_path: str,
    samed_path: str,
    boxlib_path: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    placeholder_mode: PlaceholderMode = "label",
    fix_svg_api_key: Optional[str] = None,
    fix_svg_model: Optional[str] = None,
    fix_svg_base_url: Optional[str] = None,
    fix_svg_provider: Optional[str] = None,
) -> str:
    """
    使用多模态 LLM 生成 SVG 代码

    Args:
        placeholder_mode: 占位符模式
            - "none": 无特殊样式
            - "box": 传入 boxlib 坐标
            - "label": 灰色填充+黑色边框+序号标签（推荐）
    """
    print("\n" + "=" * 60)
    print("步骤四：多模态调用生成 SVG")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    print(f"占位符模式: {placeholder_mode}")

    figure_img = Image.open(figure_path)
    samed_img = Image.open(samed_path)

    figure_width, figure_height = figure_img.size
    print(f"原图尺寸: {figure_width} x {figure_height}")

    # 基础 prompt - 让 LLM 完整重绘 SVG（文字/框线/箭头/布局全部用 SVG 元素），只有图标用占位符代替
    base_prompt = f"""You are an expert SVG coder. Your task is to faithfully recreate the ENTIRE figure as a pure, fully-editable SVG — no embedded PNG images.

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {figure_width} x {figure_height} pixels.
- Your SVG MUST use these EXACT dimensions: viewBox="0 0 {figure_width} {figure_height}" and width="{figure_width}" height="{figure_height}".
- DO NOT scale or resize the SVG.

CORE REQUIREMENT — PURE SVG (VERY IMPORTANT):
- Recreate ALL visual elements using native SVG elements: <rect>, <text>, <line>, <path>, <circle>, <polyline>, <g>, etc.
- This includes: section boxes/frames, all text labels, titles, arrows, connectors, background panels, annotation text, legends.
- Do NOT embed any PNG or raster image as a background layer (no <image href="data:...">, no base64 blobs).
- The output SVG must be fully editable — every element independently selectable and movable.

TEXT FIDELITY (CRITICAL — DO NOT VIOLATE):
- ONLY include text that is CLEARLY VISIBLE and LEGIBLE in the original image.
- Copy every text string CHARACTER FOR CHARACTER exactly as it appears. Do NOT paraphrase, translate, summarize, or reword any text.
- Do NOT invent, infer, or add any text that is not explicitly visible in the image — no titles, labels, captions, or annotations that you are guessing.
- If a text string is partially obscured or illegible, omit it entirely rather than guessing its content.
- This rule applies to ALL text elements: titles, axis labels, legends, annotations, button labels, captions, etc.

HANDLING ICONS (VERY IMPORTANT):
- The figure contains illustration icons (complex cartoon/vector icons).
- Do NOT attempt to draw these complex icons in SVG — it is impossible to reproduce them accurately.
- Instead, mark each icon area with a gray placeholder <g> block. The real icon PNG will be inserted later.
- Image 2 (the SAM3-masked image) shows exactly where each icon is — those gray/masked rectangles are the icon positions.
"""

    if placeholder_mode == "box":
        # box 模式：传入 boxlib 坐标
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_content = f.read()

        prompt_text = base_prompt + f"""
ICON COORDINATES FROM boxlib.json:
The following JSON contains precise icon coordinates detected by SAM3:
{boxlib_content}
Use these coordinates to accurately position your icon placeholders in the SVG.

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    elif placeholder_mode == "label":
        # label 模式：提供图标坐标，LLM 输出完整 SVG（文字/框线/箭头全部用 SVG 元素画出），图标位置用占位符
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_data = json.load(f)

        # 构建坐标列表，告诉 LLM 图标精确位置
        coord_list = []
        for box in boxlib_data.get("boxes", []):
            label = box.get("label", f"<AF>{box.get('id', 0) + 1:02d}")
            coord_list.append(f"  {label}: x1={box['x1']}, y1={box['y1']}, x2={box['x2']}, y2={box['y2']}")
        coords_text = "\n".join(coord_list)

        prompt_text = base_prompt + f"""
ICON PLACEHOLDER COORDINATES (VERY IMPORTANT):
For each icon listed below, place a gray placeholder <g> block at EXACTLY those coordinates.
Do NOT draw what is inside these icon areas — just place the placeholder.

{coords_text}

For EACH icon, use this placeholder format (example for <AF>01 at x1=1302, y1=199, x2=1351, y2=273):
<g id="AF01">
  <rect x="1302" y="199" width="49" height="74" fill="#808080" stroke="black" stroke-width="2"/>
  <text x="1326.5" y="236" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="14">&lt;AF&gt;01</text>
</g>

OUTPUT FORMAT:
- Output ONLY the complete SVG code, starting with <svg and ending with </svg>
- NO <image> tags, NO base64, NO embedded PNG
- All text, frames, boxes, arrows, connectors must be drawn with SVG elements
- Icon areas must use the gray placeholder <g> blocks above
- No markdown fences, no explanations"""

        print("  [label 模式] LLM 完整重绘 SVG，图标区域用占位符代替")

        contents: List[Any] = [prompt_text, figure_img, samed_img]

        print(f"发送多模态请求到: {base_url}")

        svg_code = call_llm_multimodal(
            contents=contents,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            max_tokens=65536,
        )

        if not svg_code:
            raise Exception('API 响应中没有内容，生成 SVG 失败。')

        extracted = extract_svg_code(svg_code)
        if not extracted:
            preview_head = svg_code[:500].replace('\n', '\\n')
            preview_tail = svg_code[-200:].replace('\n', '\\n') if len(svg_code) > 200 else ""
            print(f"  [调试] API 返回内容共 {len(svg_code)} 字符，未找到 <svg")
            print(f"  [调试] 内容开头: {preview_head}")
            print(f"  [调试] 内容结尾: {preview_tail}")
            raise Exception('无法从多模态 API 响应中提取有效的 SVG 代码。')
        svg_code = extracted

        print(f"  [label 模式] SVG 拼装完成，尺寸={len(svg_code)} 字节")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        svg_code = check_and_fix_svg(
            svg_code=svg_code,
            api_key=fix_svg_api_key or api_key,
            model=fix_svg_model or model,
            base_url=fix_svg_base_url or base_url,
            provider=fix_svg_provider or provider,
            output_dir=str(output_path.parent),
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_code)

        print(f"SVG 模板已保存: {output_path}")
        return str(output_path)

    else:  # none 模式
        prompt_text = base_prompt + """
Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    contents: List[Any] = [prompt_text, figure_img, samed_img]

    print(f"发送多模态请求到: {base_url}")
    
    content = call_llm_multimodal(
        contents=contents,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        max_tokens=65536,
    )

    if not content:
        raise Exception('API 响应中没有内容，生成 SVG 失败。')

    svg_code = extract_svg_code(content)

    if not svg_code:
        raise Exception('无法从多模态 API 的响应中提取出有效的 SVG 代码。')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    svg_code = check_and_fix_svg(
        svg_code=svg_code,
        api_key=fix_svg_api_key or api_key,
        model=fix_svg_model or model,
        base_url=fix_svg_base_url or base_url,
        provider=fix_svg_provider or provider,
        output_dir=str(output_path.parent),
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_code)

    print(f"SVG 模板已保存: {output_path}")
    return str(output_path)


def extract_svg_code(content: str) -> Optional[str]:
    """从响应内容中提取 SVG 代码"""
    pattern = r'(<svg[\s\S]*?</svg>)'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1)

    pattern = r'```(?:svg|xml)?\s*([\s\S]*?)```'
    match = re.search(pattern, content)
    if match:
        code = match.group(1).strip()
        if code.startswith('<svg'):
            return code

    if content.strip().startswith('<svg'):
        return content.strip()

    return None


def create_local_fallback_svg_template(
    figure_path: str,
    boxlib_path: str,
    output_path: str,
    placeholder_mode: PlaceholderMode = "label",
) -> str:
    figure_img = Image.open(figure_path)
    figure_width, figure_height = figure_img.size

    with open(boxlib_path, 'r', encoding='utf-8') as f:
        boxlib = json.load(f)

    # 基层用 samed.png（图标区域已被遮盖），避免原始图标出现在底层
    samed_path_local = str(Path(figure_path).parent / 'samed.png')
    base_img_path = samed_path_local if Path(samed_path_local).exists() else figure_path
    with open(base_img_path, 'rb') as f:
        base_img_b64 = base64.b64encode(f.read()).decode('utf-8')

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{figure_width}" height="{figure_height}" viewBox="0 0 {figure_width} {figure_height}">',
        f'<image href="data:image/png;base64,{base_img_b64}" x="0" y="0" width="{figure_width}" height="{figure_height}" />',
    ]

    if placeholder_mode in ("label", "box"):
        for box in boxlib.get("boxes", []):
            label = box.get("label", f'<AF>{box.get("id", 0) + 1:02d}')
            label_clean = label.replace('<', '').replace('>', '')
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + width / 2
            center_y = y1 + height / 2
            font_size = max(12, min(24, int(min(width, height) * 0.35)))
            svg_parts.append(
                f'<g id="{label_clean}">'
                f'<rect x="{x1}" y="{y1}" width="{width}" height="{height}" fill="#808080" stroke="black" stroke-width="2"/>'
                f'<text x="{center_x}" y="{center_y}" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="{font_size}">{escape(label)}</text>'
                f'</g>'
            )

    svg_parts.append('</svg>')
    svg_code = ''.join(svg_parts)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_code)

    print(f"本地兜底 SVG 模板已保存: {output_path}")
    return str(output_path)


# ============================================================================
# 步骤 4.5：SVG 语法验证和修复
# ============================================================================

def validate_svg_syntax(svg_code: str) -> tuple[bool, list[str]]:
    """使用 lxml 解析验证 SVG 语法"""
    try:
        from lxml import etree
        # 使用独立 XMLParser 实例，error_log 仅包含本次解析的错误，避免全局累积
        parser = etree.XMLParser(recover=False)
        try:
            etree.fromstring(svg_code.encode('utf-8'), parser)
            return True, []
        except etree.XMLSyntaxError as e:
            errors = []
            # 读 parser.error_log（仅本次解析），而非 e.error_log（全局累积）
            for error in parser.error_log:
                errors.append(f"行 {error.line}, 列 {error.column}: {error.message}")
            if not errors:
                errors.append(f"行 {e.lineno}, 列 {e.offset}: {e.msg}")
            return False, errors
    except ImportError:
        print("  警告: lxml 未安装，使用内置 xml.etree 进行验证")
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(svg_code)
            return True, []
        except ET.ParseError as e:
            return False, [f"XML 解析错误: {str(e)}"]
    except Exception as e:
        return False, [f"解析错误: {str(e)}"]


def _rule_based_svg_fix(svg_code: str) -> tuple[str, list[str]]:
    """规则修复器：在 LLM Agent 介入前，用正则自动修复常见的 XML/SVG 属性语法错误。
    
    主要处理：
    1. AttValue 错误：属性值中未转义的 & < 字符（仅限双引号属性值）
    2. 属性值内部未转义的双引号
    
    关键约束：使用 [^"<>]* 代替 (.*?) + re.DOTALL，避免跨标签边界误匹配。
    返回：(修复后的 svg_code, 应用的修复描述列表)
    """
    import re

    applied = []

    # ---- 1. 修复双引号属性值中裸露的 &（未写成 &amp;）----
    # 用 [^">]* 限定匹配范围，不会跨越 " 或 > 边界
    def _fix_amp_in_dquote_attr(m: re.Match) -> str:
        value = m.group(1)
        fixed = re.sub(r'&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', value)
        return f'="{fixed}"'

    new_code = re.sub(r'="([^">]*)"', _fix_amp_in_dquote_attr, svg_code)
    if new_code != svg_code:
        applied.append("修复双引号属性值中裸露的 & → &amp;")
        svg_code = new_code

    # ---- 2. 修复单引号属性值中裸露的 & ----
    def _fix_amp_in_squote_attr(m: re.Match) -> str:
        value = m.group(1)
        fixed = re.sub(r'&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', value)
        return f"='{fixed}'"

    new_code = re.sub(r"='([^'>]*)'", _fix_amp_in_squote_attr, svg_code)
    if new_code != svg_code:
        applied.append("修复单引号属性值中裸露的 & → &amp;")
        svg_code = new_code

    # ---- 3. 修复双引号属性值中裸露的 < ----
    def _fix_lt_in_dquote_attr(m: re.Match) -> str:
        value = m.group(1)
        fixed = value.replace('<', '&lt;')
        return f'="{fixed}"'

    new_code = re.sub(r'="([^">]*)"', _fix_lt_in_dquote_attr, svg_code)
    if new_code != svg_code:
        applied.append("修复双引号属性值中裸露的 < → &lt;")
        svg_code = new_code

    # ---- 4. 修复属性值内部未转义的双引号（在双引号属性值内出现 "）----
    # 识别模式：attrname="val"extra"val2" — 内部出现多余的 "
    # 关键约束：内嵌的 " 后面不能紧跟 \s*\w+=（那是下一个属性，不是嵌入引号）
    def _fix_unescaped_dquote(m: re.Match) -> str:
        attr_name = m.group(1)
        value = m.group(2)
        # 二次检查：value 中确实有 " 且该 " 不是下一属性的开头
        if '"' not in value:
            return m.group(0)
        # 如果每个内嵌 " 后面都紧跟 \s*\w+= 则说明是正常的相邻属性，不修复
        parts = value.split('"')
        for part in parts[1:]:  # parts[0] 是 " 前面的内容
            # 若紧接的内容看起来像 " attrname= 则跳过
            stripped = part.lstrip()
            if re.match(r'[\w][\w\-:]*\s*=', stripped):
                return m.group(0)  # 不修复，是正常属性边界
        fixed = value.replace('"', '&quot;')
        return f'{attr_name}="{fixed}"'

    new_code = re.sub(r'([\w][\w\-:]*)="([^"<>]*(?:"[^"<>]*)*)"(?=[\s/>])', _fix_unescaped_dquote, svg_code)
    if new_code != svg_code:
        applied.append("修复属性值内部未转义的双引号 → &quot;")
        svg_code = new_code

    return svg_code, applied


def fix_svg_with_llm(
    svg_code: str,
    errors: list[str],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    output_dir: str,
    max_retries: int = 8,
) -> str:
    """使用 LLM Function Calling Agent 修复 SVG 语法错误"""
    import os
    import json
    from pathlib import Path
    from openai import OpenAI
    
    print("\n  " + "-" * 50)
    print("  检测到 SVG 语法错误，启动 Agentic 修复流程...")
    print("  " + "-" * 50)

    # 1. 保存到输出目录
    temp_path = str(Path(output_dir) / "broken_template.svg")
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(svg_code)
        
    print(f"  [Agent] 已将包含错误的 SVG 保存至: {temp_path}")

    # 2. 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取文件的指定行数，会返回带有行号的代码。当你根据错误信息中的行号排查问题时，使用此工具读取报错位置前后的代码。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "文件路径"},
                        "start_line": {"type": "integer", "description": "起始行号（从1开始）"},
                        "end_line": {"type": "integer", "description": "结束行号。如果为-1表示读取到末尾"}
                    },
                    "required": ["file_path", "start_line", "end_line"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "replace_text",
                "description": "替换文件中的错误代码片段。注意：old_text必须与文件中的内容完全一致（不包含行号前缀），否则会替换失败。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "文件路径"},
                        "old_text": {"type": "string", "description": "出错的原始XML文本（切记不要带行号前缀，必须是文件里的原样内容）"},
                        "new_text": {"type": "string", "description": "修复后的正确XML文本"}
                    },
                    "required": ["file_path", "old_text", "new_text"]
                }
            }
        }
    ]

    def _read_file_tool(file_path: str, start_line: int, end_line: int) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            s = max(0, start_line - 1)
            e = len(lines) if end_line == -1 else min(len(lines), end_line)
            # 限制最多读取 50 行，防止超长行撑爆上下文
            if e - s > 50:
                e = s + 50
            
            MAX_LINE_LEN = 500  # 每行最多显示 500 字符，超长截断
            output = []
            for i in range(s, e):
                line = lines[i]
                if len(line) > MAX_LINE_LEN:
                    line = line[:MAX_LINE_LEN] + f"...[截断, 原始长度={len(lines[i])}字符]"
                output.append(f"{i+1}| {line}")
            return "\n".join(output)
        except Exception as e:
            return f"读取文件失败: {e}"

    def _strip_line_prefixes(text: str) -> str:
        """自动剥离 read_file 返回的行号前缀（如 '1| ', '23| '），Agent 有时会误带入 old_text"""
        lines = text.splitlines()
        stripped = []
        for line in lines:
            # 匹配 "数字| " 前缀格式
            import re as _re
            stripped.append(_re.sub(r'^\d+\| ?', '', line))
        return "\n".join(stripped)

    def _replace_text_tool(file_path: str, old_text: str, new_text: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if old_text not in content:
                # 尝试自动剥离行号前缀后再匹配
                stripped_old = _strip_line_prefixes(old_text)
                if stripped_old != old_text and stripped_old in content:
                    old_text = stripped_old
                    print(f"    [Tool] 自动剥离了 old_text 中的行号前缀")
                else:
                    return "错误：找不到要替换的旧文本old_text。请确保缩进、换行和内容与原文件一模一样（不要带行号前缀）。"
            new_content = content.replace(old_text, new_text)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            is_valid, new_errors = validate_svg_syntax(new_content)
            if is_valid:
                return "文件替换成功。最新验证结果：✓ 验证通过！"
            else:
                err_msg = "\n".join([f"  - {err}" for err in new_errors[:5]])
                return f"文件替换成功。但最新验证结果：仍有语法错误！\n{err_msg}"
        except Exception as e:
            return f"修改文件失败: {e}"

    # 3. 初始化 OpenAI Client
    if not base_url:
        if normalize_provider(provider) == "gemini":
            raise Exception("Fail Fast 触发: Agentic 修复流程强依赖 Function Calling。原生 Gemini SDK 暂未适配，请使用带有 base_url 的 OpenAI 兼容格式调用！")
        normalized_base_url = "https://api.openai.com/v1"
    else:
        normalized_base_url = base_url.rstrip('/')
        if not re.search(r'/v\d+$', normalized_base_url):
            normalized_base_url = f"{normalized_base_url}/v1"
        
    client = OpenAI(
        api_key=api_key,
        base_url=normalized_base_url,
        timeout=600.0,
        max_retries=0,
    )

    error_msg = "\n".join([f"  - {err}" for err in errors])
    system_prompt = f"""你是一个顶级的 XML/SVG 语法修复 Agent。
你的任务是通过使用工具来修复 SVG 文件中的语法错误。

工作流程：
1. 检查用户的错误报告和文件路径。
2. 使用 `read_file` 工具读取报错行附近的上下文（建议读取报错行前后各10行）。
3. 发现错误后，使用 `replace_text` 工具修复错误。
4. 观察 `replace_text` 的返回结果，如果提示验证通过，即可停止调用工具并回复"修复完成"；如果提示仍有错误，请继续读取并修复。

严格遵守：
- 绝对不要修改与语法错误无关的任何属性（特别是 <path> 的 d 属性、坐标数值或 Base64 数据）。
- `old_text` 必须是文件的**原始内容**，绝对不能包含 `read_file` 返回的行号前缀（如 `1| ` `2| ` 这样的前缀）。
- `read_file` 返回的每行格式是 `行号| 内容`，你只需要取 `| ` 之后的内容作为 `old_text`。
- 例如：`read_file` 返回 `5| <rect x="10"/>` ，则 `old_text` 应填 `<rect x="10"/>` 而非 `5| <rect x="10"/>`。"""

    user_prompt = f"文件路径：{temp_path}\n当前解析出来的语法错误有：\n{error_msg}\n请开始修复！"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"  [Agent] 开始循环修复流程，最大允许交互 {max_retries} 轮...")

    # 保留 system + 首条 user 消息作为固定前缀，其余动态追加
    _init_messages = messages[:]
    MAX_HISTORY_ROUNDS = 6  # 最多保留最近 6 轮工具交互，防止上下文无限膨胀

    for loop_idx in range(1, max_retries + 1):
        # 裁剪上下文：保留前 2 条初始消息 + 最近若干轮
        if len(messages) > 2 + MAX_HISTORY_ROUNDS * 2:
            messages = _init_messages[:2] + messages[-(MAX_HISTORY_ROUNDS * 2):]
        try:
            print(f"    [Round {loop_idx}] Agent 思考中...")
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0
            )
            
            response_message = completion.choices[0].message
            
            # 处理可能为空的 content
            if response_message.content is None:
                response_message.content = ""
                
            messages.append(response_message)
            
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        result = "错误：传入的参数不是合法的 JSON。"
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result})
                        continue

                    if tool_call.function.name == "read_file":
                        s_line = args.get("start_line", 1)
                        e_line = args.get("end_line", -1)
                        print(f"    [Tool] 调用 read_file: start={s_line}, end={e_line}")
                        result = _read_file_tool(args.get("file_path", temp_path), s_line, e_line)
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result})
                        
                    elif tool_call.function.name == "replace_text":
                        print(f"    [Tool] 调用 replace_text 进行修复...")
                        result = _replace_text_tool(args.get("file_path", temp_path), args.get("old_text", ""), args.get("new_text", ""))
                        print(f"    [Result] 替换结果: {result.split('。')[0]}")
                        if "验证通过" in result:
                            print(f"    [Success] SVG 语法已完全修复！")
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result})
                continue
            else:
                # Agent 认为完成了，或者没有调用 tool
                with open(temp_path, 'r', encoding='utf-8') as f:
                    final_svg = f.read()
                is_valid, final_errors = validate_svg_syntax(final_svg)
                if is_valid:
                    print("  [Agent] 修复流程正常结束，验证通过！")
                    os.remove(temp_path)
                    return final_svg
                else:
                    print(f"  [Agent] Agent 停止调用工具，但文件仍有错误: {final_errors[0]}")
                    messages.append({"role": "user", "content": f"文件仍然有错误：{final_errors[0]}，请继续调用工具修复！"})
                    continue
                    
        except Exception as e:
            err_str = str(e).lower()
            is_timeout = any(kw in err_str for kw in ("timed out", "timeout", "read timeout", "connect timeout"))
            print(f"    [Agent] 运行出错: {e}")
            if is_timeout and loop_idx < max_retries:
                print(f"    [Agent] 检测到超时，将重试 (Round {loop_idx + 1})...")
                continue
            break

    # 循环结束后的兜底检查
    with open(temp_path, 'r', encoding='utf-8') as f:
        final_svg = f.read()
    is_valid, final_errors = validate_svg_syntax(final_svg)
    
    if is_valid:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return final_svg
    
    print(f"  [Agent] 修复失败，保留错误文件用于调试: {temp_path}")
    raise Exception(f"Fail Fast 触发: Agent 修复达到最大轮数 ({max_retries})，SVG 依然存在语法错误: {final_errors[0]}")


def _detect_truncated_svg(svg_code: str, errors: list[str]) -> bool:
    """检测 SVG 是否因 LLM 输出截断导致结构不完整（如 base64 属性值未闭合）。
    
    典型症状：错误列号 >= 行长度（解析器越过了行末），且行末不以 > 或 " 结尾。
    这类错误无法通过 Agent 工具修复，需重新生成 SVG。
    """
    import re
    lines = svg_code.splitlines()
    for err in errors:
        m = re.match(r'行 (\d+), 列 (\d+):', err)
        if m:
            line_no = int(m.group(1))
            col_no = int(m.group(2))
            # 截断特征：
            # 1. 必须是 AttValue 错误（引号不匹配）
            # 2. 列号超过 100（排除 </g> 等结构错误在行末的误判）
            # 3. 列号 >= 行实际长度
            is_attvalue_err = 'AttValue' in err or 'EntityRef' in err
            if is_attvalue_err and col_no > 100 and line_no <= len(lines):
                line_len = len(lines[line_no - 1])
                if col_no >= line_len:
                    return True
    # 额外检查：SVG 中存在未闭合的 href/src 属性（含 base64）
    # 格式固定为 href="data:... 或 href='data:...，引号是属性名后面紧跟的字符
    for line in lines:
        for attr in ['href', 'src']:
            for quote in ['"', "'"]:
                pat = f'{attr}={quote}data:'
                idx = line.find(pat)
                if idx >= 0:
                    val_start = idx + len(attr) + 1  # 跳过 attr= ，指向开引号
                    closing = line.find(quote, val_start + 1)
                    if closing < 0:
                        return True  # 属性值未闭合，SVG 被截断
    return False


def check_and_fix_svg(
    svg_code: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    output_dir: str,
) -> str:
    """检查 SVG 语法并在需要时调用 LLM 修复"""
    print("\n" + "-" * 50)
    print("步骤 4.5：SVG 语法验证（使用 lxml XML 解析器）")
    print("-" * 50)

    is_valid, errors = validate_svg_syntax(svg_code)

    if is_valid:
        print("  SVG 语法验证通过！")
        return svg_code
    else:
        print(f"  发现 {len(errors)} 个语法错误")

        # 截断检测：LLM 输出被 max_tokens 截断，属性值不完整，Agent 无法修复
        if _detect_truncated_svg(svg_code, errors):
            # 无论如何先把截断的 SVG 保存，方便调试且不浪费 API 调用结果
            import os as _os
            _os.makedirs(output_dir, exist_ok=True)
            truncated_path = _os.path.join(output_dir, "truncated_template.svg")
            with open(truncated_path, 'w', encoding='utf-8') as _f:
                _f.write(svg_code)
            print(f"  [截断检测] 已保存截断的 SVG 至: {truncated_path}")
            raise Exception(
                "SVG 生成失败：LLM 输出被截断（疑似 max_tokens 不足），属性值不完整，无法通过 Agent 修复。"
                "请检查 max_tokens 参数或简化图片内容后重试。"
            )

        # 规则修复预处理：先尝试用正则自动修复 AttValue / 未转义字符等常见错误
        rule_fixed, rule_applied = _rule_based_svg_fix(svg_code)
        if rule_applied:
            print(f"  [规则修复] 应用了 {len(rule_applied)} 项规则修复:")
            for desc in rule_applied:
                print(f"    - {desc}")
            rule_valid, rule_errors = validate_svg_syntax(rule_fixed)
            if rule_valid:
                print("  [规则修复] 验证通过，无需启动 LLM Agent！")
                return rule_fixed
            else:
                print(f"  [规则修复] 规则修复后仍有 {len(rule_errors)} 个错误，继续启动 LLM Agent...")
                svg_code = rule_fixed
                errors = rule_errors
        else:
            print("  [规则修复] 无适用规则，直接启动 LLM Agent...")

        fixed_svg = fix_svg_with_llm(
            svg_code=svg_code,
            errors=errors,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            output_dir=output_dir,
        )
        return fixed_svg


# ============================================================================
# 步骤 4.7：坐标系对齐
# ============================================================================

def get_svg_dimensions(svg_code: str) -> tuple[Optional[float], Optional[float]]:
    """从 SVG 代码中提取坐标系尺寸"""
    viewbox_pattern = r'viewBox=["\']([^"\']+)["\']'
    viewbox_match = re.search(viewbox_pattern, svg_code, re.IGNORECASE)

    if viewbox_match:
        viewbox_value = viewbox_match.group(1).strip()
        parts = viewbox_value.split()
        if len(parts) >= 4:
            try:
                vb_width = float(parts[2])
                vb_height = float(parts[3])
                return vb_width, vb_height
            except ValueError:
                pass

    def parse_dimension(attr_name: str) -> Optional[float]:
        pattern = rf'{attr_name}=["\']([^"\']+)["\']'
        match = re.search(pattern, svg_code, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            numeric_match = re.match(r'([\d.]+)', value)
            if numeric_match:
                try:
                    return float(numeric_match.group(1))
                except ValueError:
                    pass
        return None

    width = parse_dimension('width')
    height = parse_dimension('height')

    if width and height:
        return width, height

    return None, None


def calculate_scale_factors(
    figure_width: int,
    figure_height: int,
    svg_width: float,
    svg_height: float,
) -> tuple[float, float]:
    """计算从 figure.png 像素坐标到 SVG 坐标的缩放因子"""
    scale_x = svg_width / figure_width
    scale_y = svg_height / figure_height
    return scale_x, scale_y


# ============================================================================
# 步骤五：图标替换到 SVG（支持序号匹配）
# ============================================================================

def replace_icons_in_svg(
    template_svg_path: str,
    icon_infos: list[dict],
    output_path: str,
    scale_factors: tuple[float, float] = (1.0, 1.0),
    match_by_label: bool = True,
) -> str:
    """
    将透明背景图标替换到 SVG 中的占位符（内联替换）。

    每个占位符 <g id="AFxx"> 直接被替换为对应的 <image> 标签，
    保持原始 SVG 结构完整可编辑。底层如存在 base64 背景图会被自动清除。

    Args:
        template_svg_path: 模板 SVG 路径
        icon_infos: 图标信息列表
        output_path: 输出路径
        scale_factors: 坐标缩放因子
        match_by_label: 是否使用序号匹配（label 模式）
    """
    print("\n" + "=" * 60)
    print("步骤五：图标替换到 SVG")
    print("=" * 60)
    print(f"匹配模式: {'序号匹配' if match_by_label else '坐标匹配'}")

    scale_x, scale_y = scale_factors
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"应用坐标缩放: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    with open(template_svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    # 防御性清除：删除可能存在的 base64 背景图片元素（LLM 有时会忽略 no-image 指令）
    svg_content = re.sub(
        r'<image\s[^>]*href=["\']data:image/[^"\']+["\'][^>]*/?>',
        '',
        svg_content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    for icon_info in icon_infos:
        label = icon_info.get("label", f"<AF>{icon_info['id'] + 1:02d}")
        label_clean = icon_info.get("label_clean", label.replace("<", "").replace(">", ""))
        nobg_path = icon_info.get("nobg_path")

        if not nobg_path or not os.path.exists(nobg_path):
            print(f"警告: 找不到图标文件 {nobg_path}")
            continue

        icon_img = Image.open(nobg_path)
        buf = io.BytesIO()
        icon_img.save(buf, format="PNG")
        icon_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        replaced = False

        if match_by_label and label:
            # 方式1：查找 id="AF01" 的 <g> 元素，直接替换为 <image>
            g_pattern = rf'<g[^>]*\bid=["\']?{re.escape(label_clean)}["\']?[^>]*>[\s\S]*?</g>'
            g_match = re.search(g_pattern, svg_content, re.IGNORECASE)

            if g_match:
                g_content = g_match.group(0)

                # 直接使用 boxlib 中的权威坐标（乘以 scale），不信任 LLM 生成的 SVG 坐标
                x = icon_info["x1"] * scale_x
                y = icon_info["y1"] * scale_y
                w = (icon_info["x2"] - icon_info["x1"]) * scale_x
                h = (icon_info["y2"] - icon_info["y1"]) * scale_y
                image_tag = (f'<image id="icon_{label_clean}" x="{x:.1f}" y="{y:.1f}" '
                             f'width="{w:.1f}" height="{h:.1f}" '
                             f'href="data:image/png;base64,{icon_b64}" '
                             f'preserveAspectRatio="xMidYMid meet"/>')
                svg_content = svg_content.replace(g_content, image_tag)
                print(f"  {label}: 替换成功 (序号匹配 <g>) at ({x:.1f}, {y:.1f}) size {w:.1f}x{h:.1f}")
                replaced = True

            # 方式2：查找包含 label 文本的 <text> 附近的 <rect>
            if not replaced:
                text_patterns = [
                    rf'<text[^>]*>[^<]*{re.escape(label)}[^<]*</text>',
                    rf'<text[^>]*>[^<]*&lt;AF&gt;{label_clean[2:]}[^<]*</text>',
                ]
                for tp in text_patterns:
                    text_match = re.search(tp, svg_content, re.IGNORECASE)
                    if text_match:
                        preceding_svg = svg_content[:text_match.start()]
                        rect_matches = list(re.finditer(r'<rect[^>]*/?\s*>', preceding_svg, re.IGNORECASE))
                        if rect_matches:
                            rect_content = rect_matches[-1].group(0)
                            x_m = re.search(r'\bx=["\']?([\d.]+)', rect_content)
                            y_m = re.search(r'\by=["\']?([\d.]+)', rect_content)
                            w_m = re.search(r'\bwidth=["\']?([\d.]+)', rect_content)
                            h_m = re.search(r'\bheight=["\']?([\d.]+)', rect_content)
                            if all([x_m, y_m, w_m, h_m]):
                                x, y = float(x_m.group(1)), float(y_m.group(1))
                                w, h = float(w_m.group(1)), float(h_m.group(1))
                                image_tag = (f'<image id="icon_{label_clean}" x="{x}" y="{y}" '
                                             f'width="{w}" height="{h}" '
                                             f'href="data:image/png;base64,{icon_b64}" '
                                             f'preserveAspectRatio="xMidYMid meet"/>')
                                svg_content = svg_content.replace(text_match.group(0), '')
                                svg_content = svg_content.replace(rect_content, image_tag, 1)
                                print(f"  {label}: 替换成功 (序号匹配 <text>) at ({x}, {y}) size {w}x{h}")
                                replaced = True
                                break

        # 回退：坐标匹配，直接替换 <rect>
        if not replaced:
            orig_x1, orig_y1 = icon_info["x1"], icon_info["y1"]
            x1 = orig_x1 * scale_x
            y1 = orig_y1 * scale_y
            w = icon_info["width"] * scale_x
            h = icon_info["height"] * scale_y
            x1_int, y1_int = int(round(x1)), int(round(y1))
            image_tag = (f'<image id="icon_{label_clean}" x="{x1:.1f}" y="{y1:.1f}" '
                         f'width="{w:.1f}" height="{h:.1f}" '
                         f'href="data:image/png;base64,{icon_b64}" '
                         f'preserveAspectRatio="xMidYMid meet"/>')

            rect_pattern = rf'<rect[^>]*x=["\']?{x1_int}(?:\.0)?["\']?[^>]*y=["\']?{y1_int}(?:\.0)?["\']?[^>]*/?\s*>'
            if re.search(rect_pattern, svg_content):
                svg_content = re.sub(rect_pattern, image_tag, svg_content, count=1)
                print(f"  {label}: 替换成功 (坐标精确匹配) at ({x1:.1f}, {y1:.1f})")
                replaced = True
            else:
                tolerance = 10
                for dx in range(-tolerance, tolerance+1, 2):
                    for dy in range(-tolerance, tolerance+1, 2):
                        rp = rf'<rect[^>]*x=["\']?{x1_int+dx}(?:\.0)?["\']?[^>]*y=["\']?{y1_int+dy}(?:\.0)?["\']?[^>]*(?:fill=["\']?(?:#[0-9A-Fa-f]{{3,6}}|gray|grey)["\']?|stroke=["\']?(?:black|#000|#000000)["\']?)[^>]*/?\s*>'
                        if re.search(rp, svg_content, re.IGNORECASE):
                            svg_content = re.sub(rp, image_tag, svg_content, count=1, flags=re.IGNORECASE)
                            print(f"  {label}: 替换成功 (坐标近似匹配) at ({x1:.1f}, {y1:.1f})")
                            replaced = True
                            break
                    if replaced:
                        break

        if not replaced:
            svg_content = svg_content.replace('</svg>', f'  {image_tag}\n</svg>')
            print(f"  {label}: 追加到 SVG at ({x1:.1f}, {y1:.1f}) (未找到匹配的占位符)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"最终 SVG 已保存: {output_path}")
    return str(output_path)


# ============================================================================
# 步骤 4.6：LLM 优化 SVG
# ============================================================================

def count_base64_images(svg_code: str) -> int:
    """统计 SVG 中嵌入的 base64 图片数量"""
    pattern = r'(?:href|xlink:href)=["\']data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    matches = re.findall(pattern, svg_code)
    return len(matches)


def validate_base64_images(svg_code: str, expected_count: int) -> tuple[bool, str]:
    """验证 SVG 中的 base64 图片是否完整"""
    actual_count = count_base64_images(svg_code)

    if actual_count < expected_count:
        return False, f"base64 图片数量不足: 期望 {expected_count}, 实际 {actual_count}"

    pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
    for match in re.finditer(pattern, svg_code):
        b64_data = match.group(1)
        if len(b64_data) % 4 != 0:
            return False, f"发现截断的 base64 数据（长度 {len(b64_data)} 不是 4 的倍数）"
        if len(b64_data) < 100:
            return False, f"发现过短的 base64 数据（长度 {len(b64_data)}），可能被截断"

    return True, f"base64 图片验证通过: {actual_count} 张图片"


def _print_cairo_installation_guide():
    """打印 Cairo 库安装指南"""
    print("\n" + "=" * 60)
    print("Cairo 库未正确安装或配置")
    print("=" * 60)
    print("\n请在 conda 环境中安装 Cairo 库：")
    print("\n  conda activate figure")
    print("  conda install -c conda-forge cairo")
    print("  pip install cairosvg")
    print("\n或者使用以下命令一键安装：")
    print("  conda install -c conda-forge cairo cairosvg")
    print("\n安装完成后重新运行程序。")
    print("=" * 60 + "\n")


def svg_to_png(svg_path: str, output_path: str, scale: float = 1.0) -> Optional[str]:
    """将 SVG 转换为 PNG"""
    try:
        # Windows 系统特殊处理：配置 Cairo DLL 路径
        if os.name == 'nt':
            dll_dirs = []
            
            # 1. 尝试当前 conda 环境
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix:
                dll_dirs.append(os.path.join(conda_prefix, 'Library', 'bin'))
            
            # 2. 尝试 sys.prefix（Python 安装目录）
            dll_dirs.append(os.path.join(sys.prefix, 'Library', 'bin'))
            
            # 3. 尝试从环境变量获取 conda 默认环境
            conda_default_env = os.environ.get('CONDA_DEFAULT_ENV')
            if conda_default_env:
                # 尝试从 CONDA_EXE 推断 conda 根目录
                conda_exe = os.environ.get('CONDA_EXE')
                if conda_exe:
                    conda_root = os.path.dirname(os.path.dirname(conda_exe))
                    env_path = os.path.join(conda_root, 'envs', conda_default_env, 'Library', 'bin')
                    if os.path.isdir(env_path):
                        dll_dirs.append(env_path)
            
            valid_dll_dirs = []
            for dll_dir in dll_dirs:
                if os.path.isdir(dll_dir):
                    # 检查是否真的包含 cairo DLL
                    cairo_dll_exists = any(
                        os.path.exists(os.path.join(dll_dir, dll_name))
                        for dll_name in ['cairo.dll', 'libcairo-2.dll', 'cairo-2.dll']
                    )
                    if cairo_dll_exists:
                        valid_dll_dirs.append(dll_dir)
                        print(f"  找到 Cairo DLL 目录: {dll_dir}")
                    
                    # 添加到 DLL 搜索路径
                    try:
                        os.add_dll_directory(dll_dir)
                    except (AttributeError, FileNotFoundError, OSError):
                        current_path = os.environ.get('PATH', '')
                        if dll_dir not in current_path.split(os.pathsep):
                            os.environ['PATH'] = dll_dir + os.pathsep + current_path
            
            if valid_dll_dirs:
                os.environ['CAIROCFFI_DLL_DIRECTORIES'] = ';'.join(dict.fromkeys(valid_dll_dirs))
            else:
                print("  警告: 未找到 Cairo DLL 文件")
        
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=output_path, scale=scale)
        return output_path
        
    except ImportError as e:
        print(f"  错误: cairosvg 未安装 ({e})")
        _print_cairo_installation_guide()
        return None
        
    except Exception as e:
        error_msg = str(e)
        
        # 检查是否是 Cairo 库缺失错误
        is_cairo_missing = 'cairo' in error_msg.lower() and ('library' in error_msg.lower() or 'dll' in error_msg.lower() or 'found' in error_msg.lower())
        
        if is_cairo_missing:
            print(f"  错误: Cairo 库未找到")
            print(f"  详细信息: {error_msg}")
        else:
            print(f"  警告: cairosvg 转换失败 ({e})")

        # 尝试 svglib + reportlab 作为 fallback（无需 Cairo）
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            print("  尝试使用 svglib + reportlab 作为备用转换器 ...")
            drawing = svg2rlg(svg_path)
            if drawing is None:
                raise ValueError("svg2rlg 返回 None，SVG 解析失败")
            if scale != 1.0:
                drawing.width *= scale
                drawing.height *= scale
                drawing.renderSVG = None
                drawing.transform = (scale, 0, 0, scale, 0, 0)
            renderPM.drawToFile(drawing, output_path, fmt="PNG")
            if os.path.exists(output_path):
                print("  ✓ svglib 转换成功")
                return output_path
        except ImportError:
            print("  svglib/reportlab 未安装，跳过此备用方案")
        except Exception as svglib_err:
            print(f"  svglib 转换失败: {svglib_err}")

        # 尝试在 conda 环境中运行 cairosvg（旧有流程，保留兼容性）
        conda_exe = os.environ.get('CONDA_EXE')
        conda_env_name = os.environ.get('CONDA_DEFAULT_ENV') or os.path.basename(sys.prefix.rstrip('\\/'))
        
        if conda_exe and conda_env_name and conda_env_name.lower() != 'base':
            try:
                print("  尝试使用独立进程运行 cairosvg ...")
                cmd = [
                    conda_exe,
                    'run',
                    '-n',
                    conda_env_name,
                    'python',
                    '-c',
                    (
                        'import cairosvg; '
                        f'cairosvg.svg2png(url={svg_path!r}, write_to={output_path!r}, scale={scale!r})'
                    ),
                ]
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(output_path):
                    print("  ✓ 独立进程转换成功")
                    return output_path
                else:
                    if result.stderr:
                        print(f"  独立进程错误: {result.stderr[:200]}")
            except Exception as subprocess_error:
                print(f"  独立进程失败: {subprocess_error}")
        
        if is_cairo_missing:
            _print_cairo_installation_guide()
        print(f"  最终错误: SVG 转 PNG 失败")
        return None


def optimize_svg_with_llm(
    figure_path: str,
    samed_path: str,
    final_svg_path: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_iterations: int = 2,
    skip_base64_validation: bool = False,
    fix_svg_api_key: Optional[str] = None,
    fix_svg_model: Optional[str] = None,
    fix_svg_base_url: Optional[str] = None,
    fix_svg_provider: Optional[str] = None,
) -> str:
    """
    使用 LLM 优化 SVG，使其与原图更加对齐

    Args:
        figure_path: 原图路径
        samed_path: 标记图路径
        final_svg_path: 输入 SVG 路径
        output_path: 输出 SVG 路径
        api_key: API Key
        model: 模型名称
        base_url: API base URL
        provider: API 提供商
        max_iterations: 最大迭代次数（0 表示跳过优化）
        skip_base64_validation: 是否跳过 base64 图片验证

    Returns:
        优化后的 SVG 路径
    """
    print("\n" + "=" * 60)
    print("步骤 4.6：LLM 优化 SVG（位置和样式对齐）")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"模型: {model}")
    print(f"最大迭代次数: {max_iterations}")

    # 如果迭代次数为 0，直接复制文件并跳过优化
    if max_iterations == 0:
        print("  迭代次数为 0，跳过 LLM 优化")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final_svg_path, output_path)
        print(f"  直接复制模板: {final_svg_path} -> {output_path}")
        return str(output_path)

    with open(final_svg_path, 'r', encoding='utf-8') as f:
        current_svg = f.read()

    output_dir = Path(final_svg_path).parent

    original_image_count = 0
    if not skip_base64_validation:
        original_image_count = count_base64_images(current_svg)
        print(f"原始 SVG 包含 {original_image_count} 张嵌入图片")
    else:
        print("跳过 base64 图片验证（模板 SVG）")

    for iteration in range(max_iterations):
        print(f"\n  优化迭代 {iteration + 1}/{max_iterations}")
        print("  " + "-" * 50)

        current_svg_path = output_dir / f"temp_svg_iter_{iteration}.svg"
        current_png_path = output_dir / f"temp_png_iter_{iteration}.png"

        with open(current_svg_path, 'w', encoding='utf-8') as f:
            f.write(current_svg)

        png_result = svg_to_png(str(current_svg_path), str(current_png_path))

        if png_result is None:
            print("  无法将 SVG 转换为 PNG，跳过优化")
            break

        figure_img = Image.open(figure_path)
        samed_img = Image.open(samed_path)
        current_png_img = Image.open(str(current_png_path))

        prompt = f"""You are an expert SVG optimizer. Compare the current SVG rendering with the original figure and optimize the SVG code to better match the original.

I'm providing you with 4 inputs:
1. **Image 1 (figure.png)**: The original target figure that we want to replicate
2. **Image 2 (samed.png)**: The same figure with icon positions marked as gray rectangles with labels (<AF>01, <AF>02, etc.)
3. **Image 3 (current SVG rendered as PNG)**: The current state of our SVG
4. **Current SVG code**: The SVG code that needs optimization

Please carefully compare and check the following **THREE MAJOR ASPECTS**:

## ASPECT 1: POSITION (位置)
1. **Icons (图标)**: Are icon placeholder positions matching the original?
2. **Text (文字)**: Are text elements positioned correctly?
3. **Arrows (箭头)**: Are arrows starting/ending at correct positions?
4. **Lines/Borders (线条)**: Are lines and borders aligned properly?

## ASPECT 2: STYLE (样式)
5. **Icons (图标)**: Icon placeholder sizes, proportions (must have gray fill #808080, black border, and centered label)
6. **Text (文字)**: Font sizes, colors, weights
7. **Arrows (箭头)**: Arrow styles, thicknesses, colors
8. **Lines/Borders (线条)**: Line styles, colors, stroke widths

## ASPECT 3: TEXT FIDELITY AUDIT (文字忠实度审核) — CRITICAL
Compare EVERY `<text>` element in the current SVG against what is clearly visible in Image 1 (the original figure):
9. **Remove hallucinated text**: Delete any `<text>` whose content does NOT appear in the original image (invented titles, labels, captions, annotations).
10. **Correct wrong text**: Fix any `<text>` whose content is paraphrased, translated, summarized, or otherwise different from the original — copy the exact characters visible in the image.
11. **Omit illegible text**: If the original text is partially obscured or too small to read clearly, remove that `<text>` element rather than guessing.
12. **Do NOT add new text**: Do not introduce any `<text>` element whose string is not unambiguously visible in Image 1.

**CURRENT SVG CODE:**
```xml
{current_svg}
```

**IMPORTANT:**
- Output ONLY the optimized SVG code
- Start with <svg and end with </svg>
- Do NOT include markdown formatting or explanations
- Keep all icon placeholder structures intact (the <g> elements with id like "AF01")
- Focus on position, style corrections, AND text fidelity — all three aspects must be addressed in this single pass
- TEXT RULE: every `<text>` in the output must correspond to text clearly visible in the original image, copied character-for-character"""

        contents = [prompt, figure_img, samed_img, current_png_img]

        try:
            print("  发送优化请求...")
            content = call_llm_multimodal(
                contents=contents,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=50000,
                temperature=0.3,
            )

            if not content:
                print("  响应为空")
                continue

            optimized_svg = extract_svg_code(content)

            if not optimized_svg:
                print("  无法从响应中提取 SVG 代码")
                continue

            is_valid, errors = validate_svg_syntax(optimized_svg)

            if not is_valid:
                print(f"  优化后的 SVG 有语法错误，尝试修复...")
                optimized_svg = fix_svg_with_llm(
                    svg_code=optimized_svg,
                    errors=errors,
                    api_key=fix_svg_api_key or api_key,
                    model=fix_svg_model or model,
                    base_url=fix_svg_base_url or base_url,
                    provider=fix_svg_provider or provider,
                )

            if not skip_base64_validation:
                images_valid, images_msg = validate_base64_images(optimized_svg, original_image_count)
                if not images_valid:
                    print(f"  警告: {images_msg}")
                    print("  拒绝此次优化，保留上一版本 SVG")
                    continue
                print(f"  {images_msg}")

            current_svg = optimized_svg
            print("  优化迭代完成")

        except Exception as e:
            print(f"  优化过程出错: {e}")
            continue

        try:
            current_svg_path.unlink(missing_ok=True)
            current_png_path.unlink(missing_ok=True)
        except:
            pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(current_svg)

    final_png_path = output_path.with_suffix('.png')
    svg_to_png(str(output_path), str(final_png_path))
    print(f"\n  优化后的 SVG 已保存: {output_path}")
    print(f"  PNG 预览已保存: {final_png_path}")

    return str(output_path)


# 主函数：完整流程
# ============================================================================

def method_to_svg(
    method_text: str,
    output_dir: str = "./output",
    resume_dir: Optional[str] = None,
    input_image: Optional[str] = None,
    sam_prompts: str = "icon, illustration, person, robot, machine, device, animal, Signal, MRI",
    min_score: float = 0.2,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
    rmbg_model_path: Optional[str] = None,
    stop_after: int = 5,
    placeholder_mode: PlaceholderMode = "label",
    optimize_iterations: int = 2,
    merge_threshold: float = 0.3,
    max_box_area_ratio: float = 0.1,
    enable_enhanced_detection: bool = True,
    image_api_key: Optional[str] = None,
    image_base_url: Optional[str] = None,
    image_provider: Optional[str] = "gemini",
    image_model: Optional[str] = None,
    svg_api_key: Optional[str] = None,
    svg_base_url: Optional[str] = None,
    svg_provider: Optional[str] = "gemini",
    svg_model: Optional[str] = None,
    fix_svg_api_key: Optional[str] = None,
    fix_svg_base_url: Optional[str] = None,
    fix_svg_provider: Optional[str] = "gemini",
    fix_svg_model: Optional[str] = None,
) -> dict:
    """
    完整流程：Paper Method → SVG with Icons

    Args:
        method_text: Paper method 文本内容
        output_dir: 输出目录
        sam_prompts: SAM3 文本提示，支持逗号分隔的多个prompt（如 "icon,diagram,arrow"）
        min_score: SAM3 最低置信度
        sam_backend: SAM3 后端（local/fal/roboflow/api）
        sam_api_key: SAM3 API Key（api 模式使用）
        sam_max_masks: SAM3 API 最大 masks 数（api 模式使用）
        rmbg_model_path: RMBG 模型本地路径（可选）
        stop_after: 调试用，在第几步后停止
        placeholder_mode: "label" | "box" | "empty"
        optimize_iterations: 优化 SVG 的次数
        merge_threshold: Box合并阈值，重叠比例超过此值则合并（0表示不合并，默认0.9）
        image_api_key: 生图链路 API Key
        image_base_url: 生图链路 API base URL
        image_provider: 生图链路 provider
        image_model: 生图链路模型
        svg_api_key: SVG链路 API Key
        svg_base_url: SVG链路 API base URL
        svg_provider: SVG链路 provider
        svg_model: SVG链路模型
        fix_svg_api_key: SVG修复链路 API Key
        fix_svg_base_url: SVG修复链路 API base URL
        fix_svg_provider: SVG修复链路 provider
        fix_svg_model: SVG修复链路模型
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 如果提供了 resume_dir，将已有的中间产物复制到当前输出目录，以便 skip 逻辑生效
    if resume_dir:
        import shutil as _shutil
        resume_path = Path(resume_dir)
        if resume_path.exists():
            print(f"  [Resume] 从 {resume_path} 复制已有产物...")
            # 复制单文件产物
            for fname in ["figure.png", "samed.png", "boxlib.json"]:
                src = resume_path / fname
                dst = out_path / fname
                if src.exists() and not dst.exists():
                    _shutil.copy2(str(src), str(dst))
                    print(f"  [Resume] 复制: {fname}")
            # 复制 icons/ 目录
            src_icons = resume_path / "icons"
            dst_icons = out_path / "icons"
            if src_icons.exists() and not dst_icons.exists():
                _shutil.copytree(str(src_icons), str(dst_icons))
                print(f"  [Resume] 复制: icons/ ({len(list(src_icons.glob('*.png')))} 文件)")
        else:
            print(f"  [Resume] 警告: resume_dir 不存在: {resume_path}，忽略")

    # 1. 解析模型配置
    # 如果提供了 input_image，则跳过图像生成，不需要 image 链路的 API key
    _skip_image_gen = bool(input_image and Path(input_image).is_file())
    if _skip_image_gen:
        image_config = {"provider": "none", "api_key": "", "base_url": "", "model": ""}
    else:
        image_config = resolve_llm_config(
            provider=image_provider,
            api_key=image_api_key,
            base_url=image_base_url,
            model=image_model,
            model_kind="image"
        )
    svg_config = resolve_llm_config(
        provider=svg_provider,
        api_key=svg_api_key,
        base_url=svg_base_url,
        model=svg_model,
        model_kind="svg"
    )
    fix_svg_config = resolve_llm_config(
        provider=fix_svg_provider,
        api_key=fix_svg_api_key,
        base_url=fix_svg_base_url,
        model=fix_svg_model,
        model_kind="fix_svg"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Paper Method 到 SVG 图标替换流程 (Label 模式增强版 + Box合并)")
    print("=" * 60)
    print(f"Image Provider: {image_config['provider']}")
    print(f"SVG Provider: {svg_config['provider']}")
    print(f"输出目录: {output_dir}")
    print(f"生图模型: {image_config['model']}")
    print(f"SVG模型: {svg_config['model']}")
    print(f"SAM提示词: {sam_prompts}")
    print(f"最低置信度: {min_score}")
    sam_backend_value = "fal" if sam_backend == "api" else sam_backend
    print(f"SAM后端: {sam_backend_value}")
    if sam_backend_value == "fal":
        print(f"SAM3 API max_masks: {sam_max_masks}")
    print(f"执行到步骤: {stop_after}")
    print(f"占位符模式: {placeholder_mode}")
    print(f"优化迭代次数: {optimize_iterations}")
    print(f"Box合并阈值: {merge_threshold}")
    print(f"最大框面积上限: {'不限制' if max_box_area_ratio <= 0 else f'{max_box_area_ratio:.1%}'}")
    print("=" * 60)

    # 步骤一：生成图片（如果提供了 input_image 则直接复制；若 figure.png 已存在则跳过）
    figure_path = output_dir / "figure.png"
    if input_image and Path(input_image).is_file() and not figure_path.exists():
        import shutil as _shutil2
        _shutil2.copy2(input_image, str(figure_path))
        print("\n" + "=" * 60)
        print(f"步骤一：[跳过] 使用上传图片 {input_image}，已复制到 {figure_path}")
        print("=" * 60)
    elif figure_path.exists():
        print("\n" + "=" * 60)
        print("步骤一：[跳过] 检测到 figure.png 已存在，跳过图片生成步骤")
        print("=" * 60)
    else:
        generate_figure_from_method(
            method_text=method_text,
            output_path=str(figure_path),
            api_key=image_config["api_key"],
            model=image_config["model"],
            base_url=image_config["base_url"],
            provider=image_config["provider"],
        )

    if stop_after == 1:
        print("\n" + "=" * 60)
        print("已在步骤 1 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": None,
            "boxlib_path": None,
            "icon_infos": [],
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤二：SAM3 分割（如果 samed.png + boxlib.json 已存在则跳过）
    samed_path_candidate = output_dir / "samed.png"
    boxlib_path_candidate = output_dir / "boxlib.json"
    if samed_path_candidate.exists() and boxlib_path_candidate.exists():
        print("\n" + "=" * 60)
        print("步骤二：[跳过] 检测到 samed.png + boxlib.json 已存在，跳过 SAM3 分割步骤")
        print("=" * 60)
        samed_path = str(samed_path_candidate)
        boxlib_path = str(boxlib_path_candidate)
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_data = json.load(f)
        valid_boxes = boxlib_data.get("boxes", [])
        print(f"  从缓存读取到 {len(valid_boxes)} 个 boxes")
    else:
        samed_path, boxlib_path, valid_boxes = segment_with_sam3(
            image_path=str(figure_path),
            output_dir=str(output_dir),
            text_prompts=sam_prompts,
            min_score=min_score,
            merge_threshold=merge_threshold,
            max_box_area_ratio=max_box_area_ratio,
            sam_backend=sam_backend_value,
            sam_api_key=sam_api_key,
            sam_max_masks=sam_max_masks,
            enable_enhanced_detection=enable_enhanced_detection,
        )

    if len(valid_boxes) == 0:
        print("\n警告: 没有检测到有效的图标，将继续使用原始图像生成 SVG（无图标替换）")
    else:
        print(f"\n检测到 {len(valid_boxes)} 个图标")

    if stop_after == 2:
        print("\n" + "=" * 60)
        print("已在步骤 2 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": [],
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤三：裁切 + 去背景（如果 icons/ 文件夹已存在 nobg 文件则跳过）
    icons_dir_candidate = output_dir / "icons"
    nobg_files = list(icons_dir_candidate.glob("*_nobg.png")) if icons_dir_candidate.exists() else []
    if nobg_files:
        print("\n" + "=" * 60)
        print(f"步骤三：[跳过] 检测到 icons/ 已存在 {len(nobg_files)} 个 _nobg.png 文件，跳过 RMBG 步骤")
        print("=" * 60)
        # 从文件系统重建 icon_infos
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_data = json.load(f)
        icon_infos = []
        for box_info in boxlib_data.get("boxes", []):
            box_id = box_info["id"]
            label = box_info.get("label", f"<AF>{box_id + 1:02d}")
            label_clean = label.replace("<", "").replace(">", "")
            x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]
            crop_path = icons_dir_candidate / f"icon_{label_clean}.png"
            nobg_path = icons_dir_candidate / f"icon_{label_clean}_nobg.png"
            icon_infos.append({
                "id": box_id,
                "label": label,
                "label_clean": label_clean,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
                "crop_path": str(crop_path),
                "nobg_path": str(nobg_path),
            })
            print(f"  {label}: 使用缓存 -> {nobg_path}")
    else:
        icon_infos = crop_and_remove_background(
            image_path=str(figure_path),
            boxlib_path=boxlib_path,
            output_dir=str(output_dir),
            rmbg_model_path=rmbg_model_path,
        )

    if stop_after == 3:
        print("\n" + "=" * 60)
        print("已在步骤 3 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # 步骤四：生成 SVG 模板
    template_svg_path = output_dir / "template.svg"
    generate_svg_template(
        figure_path=str(figure_path),
        samed_path=samed_path,
        boxlib_path=boxlib_path,
        output_path=str(template_svg_path),
        api_key=svg_config["api_key"],
        model=svg_config["model"],
        base_url=svg_config["base_url"],
        provider=svg_config["provider"],
        placeholder_mode=placeholder_mode,
        fix_svg_api_key=fix_svg_config["api_key"],
        fix_svg_model=fix_svg_config["model"],
        fix_svg_base_url=fix_svg_config["base_url"],
        fix_svg_provider=fix_svg_config["provider"],
    )

    # 步骤 4.6：LLM 优化 SVG 模板（可配置迭代次数，0 表示跳过）
    optimized_template_path = output_dir / "optimized_template.svg"
    optimize_svg_with_llm(
        figure_path=str(figure_path),
        samed_path=samed_path,
        final_svg_path=str(template_svg_path),
        output_path=str(optimized_template_path),
        api_key=svg_config["api_key"],
        model=svg_config["model"],
        base_url=svg_config["base_url"],
        provider=svg_config["provider"],
        max_iterations=optimize_iterations,
        skip_base64_validation=True,
        fix_svg_api_key=fix_svg_config["api_key"],
        fix_svg_model=fix_svg_config["model"],
        fix_svg_base_url=fix_svg_config["base_url"],
        fix_svg_provider=fix_svg_config["provider"],
    )

    if stop_after == 4:
        print("\n" + "=" * 60)
        print("已在步骤 4 后停止")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
            "template_svg_path": str(template_svg_path),
            "optimized_template_path": str(optimized_template_path),
            "final_svg_path": None,
        }

    # 步骤 4.7：坐标系对齐
    print("\n" + "-" * 50)
    print("步骤 4.7：坐标系对齐")
    print("-" * 50)

    figure_img = Image.open(figure_path)
    figure_width, figure_height = figure_img.size
    print(f"原图尺寸: {figure_width} x {figure_height}")

    with open(optimized_template_path, 'r', encoding='utf-8') as f:
        svg_code = f.read()

    svg_width, svg_height = get_svg_dimensions(svg_code)

    if svg_width and svg_height:
        print(f"SVG 尺寸: {svg_width} x {svg_height}")

        if abs(svg_width - figure_width) < 1 and abs(svg_height - figure_height) < 1:
            print("尺寸匹配，使用 1:1 坐标映射")
            scale_factors = (1.0, 1.0)
        else:
            scale_x, scale_y = calculate_scale_factors(
                figure_width, figure_height, svg_width, svg_height
            )
            scale_factors = (scale_x, scale_y)
            print(f"尺寸不匹配，计算缩放因子: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    else:
        print("警告: 无法提取 SVG 尺寸，使用 1:1 坐标映射")
        scale_factors = (1.0, 1.0)

    # 步骤五：图标替换
    final_svg_path = output_dir / "final.svg"
    replace_icons_in_svg(
        template_svg_path=str(optimized_template_path),
        icon_infos=icon_infos,
        output_path=str(final_svg_path),
        scale_factors=scale_factors,
        match_by_label=(placeholder_mode == "label"),
    )

    print("\n" + "=" * 60)
    print("流程完成！")
    print("=" * 60)
    print(f"原始图片: {figure_path}")
    print(f"标记图片: {samed_path}")
    print(f"Box信息: {boxlib_path}")
    print(f"图标数量: {len(icon_infos)}")
    print(f"SVG模板: {template_svg_path}")
    print(f"优化后模板: {optimized_template_path}")
    print(f"最终SVG: {final_svg_path}")

    return {
        "figure_path": str(figure_path),
        "samed_path": samed_path,
        "boxlib_path": boxlib_path,
        "icon_infos": icon_infos,
        "template_svg_path": str(template_svg_path),
        "optimized_template_path": str(optimized_template_path),
        "final_svg_path": str(final_svg_path),
    }


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper Method 到 SVG 图标替换工具 (Label 模式增强版 + Box合并)"
    )

    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--method_text", help="Paper method 文本内容")
    input_group.add_argument("--method_file", default="./paper.txt", help="包含 paper method 的文本文件路径")

    # 输出参数
    parser.add_argument("--output_dir", default="./output", help="输出目录（默认: ./output）")

    # 模型参数
    parser.add_argument("--image_model", default=None, help="生图模型")
    parser.add_argument("--svg_model", default=None, help="SVG生成模型")
    parser.add_argument("--image_provider", default="gemini", help="生图链路 provider")
    parser.add_argument("--image_api_key", default=None, help="生图链路 API Key")
    parser.add_argument("--image_base_url", default=None, help="生图链路 API base URL")
    parser.add_argument("--svg_provider", default="gemini", help="SVG链路 provider")
    parser.add_argument("--svg_api_key", default=None, help="SVG链路 API Key")
    parser.add_argument("--svg_base_url", default=None, help="SVG链路 API base URL")
    parser.add_argument("--fix_svg_model", default=None, help="SVG修复模型")
    parser.add_argument("--fix_svg_provider", default="gemini", help="SVG修复链路 provider")
    parser.add_argument("--fix_svg_api_key", default=None, help="SVG修复链路 API Key")
    parser.add_argument("--fix_svg_base_url", default=None, help="SVG修复链路 API base URL")

    # Step 1 参考图片参数
    parser.add_argument(
        "--use_reference_image",
        action="store_true",
        help="步骤一使用参考图片风格（需要同时提供 --reference_image_path）"
    )
    parser.add_argument("--reference_image_path", default=None, help="参考图片路径（可选）")
    parser.add_argument("--input_image", default=None, help="直接提供输入图片路径，跳过步骤一的图片生成（使用上传图片代替AI生成）")
    parser.add_argument("--resume_dir", default=None, help="从上一次运行的输出目录续跑：自动复制 figure.png/samed.png/boxlib.json/icons/ 到新目录以跳过已完成的步骤")

    # SAM3 参数
    parser.add_argument("--sam_prompt", default="icon, illustration, person, robot, machine, device, animal, Signal, MRI", help="SAM3 文本提示，支持逗号分隔多个prompt")
    parser.add_argument("--min_score", type=float, default=0.2, help="SAM3 最低置信度阈值（默认: 0.2）")
    parser.add_argument(
        "--sam_backend",
        choices=["local", "fal", "roboflow", "api"],
        default="local",
        help="SAM3 后端：local(本地部署)/fal(fal.ai)/roboflow(Roboflow)/api(旧别名=fal)",
    )
    parser.add_argument("--sam_api_key", default=None, help="SAM3 API Key（默认使用 FAL_KEY）")
    parser.add_argument(
        "--sam_max_masks",
        type=int,
        default=32,
        help="SAM3 API 最大 masks 数（仅 api 后端，默认: 32）",
    )

    # RMBG 参数
    parser.add_argument("--rmbg_model_path", default=None, help="RMBG 模型本地路径（可选）")

    # 流程控制参数
    parser.add_argument(
        "--stop_after",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="执行到指定步骤后停止（1-5，默认: 5 完整流程）"
    )

    # 占位符模式参数
    parser.add_argument(
        "--placeholder_mode",
        choices=["none", "box", "label"],
        default="label",
        help="占位符模式：none(无样式)/box(传坐标)/label(序号匹配)（默认: label）"
    )

    # 步骤 4.6 优化迭代次数参数
    parser.add_argument(
        "--optimize_iterations",
        type=int,
        default=0,
        help="步骤 4.6 LLM 优化迭代次数（0 表示跳过优化，默认: 0）"
    )

    # Box 合并阈值参数
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=0.3,
        help="Box合并阈值，重叠比例超过此值则合并（0表示不合并，默认: 0.3）"
    )
    parser.add_argument(
        "--max_box_area_ratio",
        type=float,
        default=0.1,
        help="最大框面积上限，merge 与最终 box 过滤共用（0表示不限制，默认: 0.1）"
    )

    # 增强检测参数
    parser.add_argument(
        "--enable_enhanced_detection",
        action="store_true",
        help="启用增强检测补充SAM3遗漏的视觉元素"
    )

    args = parser.parse_args()

    if args.use_reference_image and not args.reference_image_path:
        parser.error("--use_reference_image 需要 --reference_image_path")
    if args.reference_image_path and not Path(args.reference_image_path).is_file():
        parser.error(f"参考图片不存在: {args.reference_image_path}")

    USE_REFERENCE_IMAGE = bool(args.use_reference_image)
    REFERENCE_IMAGE_PATH = args.reference_image_path
    if REFERENCE_IMAGE_PATH:
        USE_REFERENCE_IMAGE = True

    # 获取 method 文本：优先使用 --method_text
    method_text = args.method_text
    if method_text is None:
        with open(args.method_file, 'r', encoding='utf-8') as f:
            method_text = f.read()

    # 运行完整流程
    result = method_to_svg(
        method_text=method_text,
        output_dir=args.output_dir,
        resume_dir=args.resume_dir,
        input_image=args.input_image,
        sam_prompts=args.sam_prompt,
        min_score=args.min_score,
        sam_backend=args.sam_backend,
        sam_api_key=args.sam_api_key,
        sam_max_masks=args.sam_max_masks,
        rmbg_model_path=args.rmbg_model_path,
        stop_after=args.stop_after,
        placeholder_mode=args.placeholder_mode,
        optimize_iterations=args.optimize_iterations,
        merge_threshold=args.merge_threshold,
        max_box_area_ratio=args.max_box_area_ratio,
        enable_enhanced_detection=args.enable_enhanced_detection,
        image_api_key=args.image_api_key,
        image_base_url=args.image_base_url,
        image_provider=args.image_provider,
        image_model=args.image_model,
        svg_api_key=args.svg_api_key,
        svg_base_url=args.svg_base_url,
        svg_provider=args.svg_provider,
        svg_model=args.svg_model,
        fix_svg_api_key=args.fix_svg_api_key,
        fix_svg_base_url=args.fix_svg_base_url,
        fix_svg_provider=args.fix_svg_provider,
        fix_svg_model=args.fix_svg_model,
    )
