"""
AstrBot 消息防抖插件
使用 BERT 模型判断用户是否说完一句话，避免频繁调用 LLM
"""

import os
import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from astrbot.api import logger


@dataclass
class MessageBuffer:
    """消息缓冲区，用于存储用户未完成的消息"""
    messages: list = field(default_factory=list)
    last_update: float = field(default_factory=time.time)
    
    def add(self, message: str):
        self.messages.append(message)
        self.last_update = time.time()
    
    def get_full_text(self) -> str:
        return " ".join(self.messages)
    
    def clear(self):
        self.messages = []
        self.last_update = time.time()
    
    def is_timeout(self, timeout_seconds: int) -> bool:
        if timeout_seconds <= 0:
            return False
        return time.time() - self.last_update > timeout_seconds


class SentenceClassifier:
    """句子完整性分类器"""
    
    def __init__(self, model_path: str, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.session = ort.InferenceSession(model_path)
        logger.info(f"🚀 消息防抖模型已加载: {model_path}")
    
    def predict(self, text: str) -> tuple[bool, float]:
        """
        预测句子是否完整
        返回: (是否完整, 完整概率)
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="np", 
            padding=True, 
            truncation=True,
            max_length=64
        )
        
        outputs = self.session.run(
            output_names=["logits"],
            input_feed={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        )
        
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定的 softmax
        softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        score_send = float(softmax_probs[0][1])  # Label 1 = SEND
        return score_send, score_send



class DebouncePlugin(Star):
    
    def __init__(self, context: Context):
        super().__init__(context)
        
        # 消息缓冲区 {session_id: MessageBuffer}
        self.buffers: Dict[str, MessageBuffer] = {}
        
        # 分类器（延迟加载）
        self.classifier: Optional[SentenceClassifier] = None
        
        # 插件目录
        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 超时检查任务
        self._timeout_task: Optional[asyncio.Task] = None
    
    def _get_config(self, key: str, default=None):
        """获取插件配置"""
        config = self.context.get_config()
        return config.get(key, default)
    
    def _load_classifier(self):
        """加载分类模型"""
        if self.classifier is not None:
            return
        
        model_type = self._get_config("model_type", "small")
        model_dir = os.path.join(self.plugin_dir, "models", model_type)
        model_path = os.path.join(model_dir, "model.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        
        if not os.path.exists(model_path):
            logger.error(f"❌ 模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        if not os.path.exists(tokenizer_path):
            logger.error(f"❌ Tokenizer 不存在: {tokenizer_path}")
            raise FileNotFoundError(f"Tokenizer 不存在: {tokenizer_path}")
        
        self.classifier = SentenceClassifier(model_path, tokenizer_path)
        logger.info(f"✅ 消息防抖插件已加载模型: {model_type}")
    
    def _get_buffer(self, session_id: str) -> MessageBuffer:
        """获取或创建消息缓冲区"""
        if session_id not in self.buffers:
            self.buffers[session_id] = MessageBuffer()
        return self.buffers[session_id]
    
    @filter.on_llm_request(priority=100)  # 高优先级，优先拦截
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """LLM 请求前的钩子 - 核心防抖逻辑"""
        
        # 检查是否启用
        if not self._get_config("enabled", True):
            return
        
        # 延迟加载模型
        try:
            self._load_classifier()
        except FileNotFoundError as e:
            logger.warning(f"消息防抖插件模型加载失败，跳过防抖: {e}")
            return
        
        # 启动超时检查任务（懒加载）
        if self._timeout_task is None:
            self._timeout_task = asyncio.create_task(self._timeout_checker())
        
        session_id = event.message_obj.session_id
        message_text = event.message_str.strip()
        
        if not message_text:
            return
        
        buffer = self._get_buffer(session_id)
        threshold = self._get_config("send_threshold", 0.5)
        timeout_seconds = self._get_config("timeout_seconds", 30)
        debug_mode = self._get_config("debug_mode", False)
        
        # 检查是否超时
        if buffer.messages and buffer.is_timeout(timeout_seconds):
            # 超时，合并已有消息并发送
            full_text = buffer.get_full_text() + " " + message_text
            buffer.clear()
            
            # 修改请求内容为完整消息
            if hasattr(req, 'prompt') and req.prompt:
                req.prompt = full_text
            
            if debug_mode:
                logger.info(f"[防抖] 超时发送: {full_text}")
            return
        
        # 添加新消息到缓冲区
        buffer.add(message_text)
        full_text = buffer.get_full_text()
        
        # 模型预测
        score_send, _ = self.classifier.predict(full_text)
        is_complete = score_send >= threshold
        
        if debug_mode:
            logger.info(f"[防抖] 文本: '{full_text}' | 完整概率: {score_send:.4f} | 判定: {'发送' if is_complete else '等待'}")
        
        if is_complete:
            # 句子完整，清空缓冲区并发送
            buffer.clear()
            
            # 修改请求内容为完整消息
            if hasattr(req, 'prompt') and req.prompt:
                req.prompt = full_text
            
            if debug_mode:
                logger.info(f"[防抖] 完整发送: {full_text}")
        else:
            # 句子未完整，阻止请求，停止事件传播
            event.stop_event()
            
            if debug_mode:
                logger.info(f"[防抖] 等待更多输入，当前缓存: {full_text}")
    
    async def _timeout_checker(self):
        """后台任务：定期检查并清理超时的缓冲区"""
        while True:
            await asyncio.sleep(10)  # 每10秒检查一次
            
            timeout_seconds = self._get_config("timeout_seconds", 30)
            if timeout_seconds <= 0:
                continue
            
            # 清理超时的缓冲区
            expired_sessions = []
            for session_id, buffer in self.buffers.items():
                if buffer.messages and buffer.is_timeout(timeout_seconds * 2):
                    # 超过2倍超时时间的缓冲区直接清理
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.buffers[session_id]
                logger.debug(f"[防抖] 清理过期缓冲区: {session_id}")
    
    async def terminate(self):
        """插件卸载时的清理"""
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
        
        self.buffers.clear()
        self.classifier = None
        logger.info("🛑 消息防抖插件已卸载")
