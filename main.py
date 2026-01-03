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
        
        # 正在等待 LLM 回复的会话集合
        self.pending_llm_sessions: set = set()
        
        # 需要丢弃回复的会话集合
        self.discard_responses: set = set()
        
        # 记录上次发送的消息内容（用于恢复）
        self.last_sent_messages: Dict[str, str] = {}
        
        # 正在等待的会话集合（防止重复等待）
        self.waiting_sessions: set = set()
        
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
        
        # 检查模型是否存在，不存在则自动下载
        if not os.path.exists(model_path):
            logger.info(f"模型文件不存在，尝试从 ModelScope 下载: {model_type}")
            if not self._download_model_from_modelscope(model_type, model_dir):
                logger.error(f"❌ 模型下载失败: {model_type}")
                raise FileNotFoundError(f"模型文件不存在且下载失败: {model_path}")
        
        if not os.path.exists(tokenizer_path):
            logger.error(f"❌ Tokenizer 不存在: {tokenizer_path}")
            raise FileNotFoundError(f"Tokenizer 不存在: {tokenizer_path}")
        
        self.classifier = SentenceClassifier(model_path, tokenizer_path)
        logger.info(f"✅ 消息防抖插件已加载模型: {model_type}")
    
    def _download_model_from_modelscope(self, model_type: str, target_dir: str) -> bool:
        """从 ModelScope 下载模型"""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            
            # ModelScope 模型仓库映射
            model_repos = {
                "small": "advent259141/astrbot_debouncer_small",
                "normal": "advent259141/astrbot_debouncer_normal"
            }
            
            repo_id = model_repos.get(model_type)
            if not repo_id:
                logger.warning(f"模型类型 {model_type} 无需下载")
                return False
            
            logger.info(f"🔄 正在从 ModelScope 下载模型: {repo_id}")
            
            # 下载到临时目录
            cache_dir = snapshot_download(
                repo_id,
                cache_dir=os.path.join(self.plugin_dir, ".cache")
            )
            
            # 复制文件到目标目录
            import shutil
            os.makedirs(target_dir, exist_ok=True)
            
            # 复制模型文件
            src_model = os.path.join(cache_dir, "model", "model.onnx")
            if os.path.exists(src_model):
                shutil.copy2(src_model, os.path.join(target_dir, "model.onnx"))
                logger.info("✅ 模型文件下载完成")
            
            # 复制 tokenizer 目录
            src_tokenizer = os.path.join(cache_dir, "tokenizer")
            dst_tokenizer = os.path.join(target_dir, "tokenizer")
            if os.path.exists(src_tokenizer):
                shutil.copytree(src_tokenizer, dst_tokenizer, dirs_exist_ok=True)
                logger.info("✅ Tokenizer 文件下载完成")
            
            logger.info(f"🎉 模型 {model_type} 下载成功")
            return True
            
        except ImportError:
            logger.error("❌ modelscope 库未安装，请运行: pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"❌ 从 ModelScope 下载模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
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
        cancel_on_new = self._get_config("cancel_on_new_message", True)
        
        # 检查该会话是否正在等待 LLM 回复
        if cancel_on_new and session_id in self.pending_llm_sessions:
            # 用户在等待回复时又发了新消息，说明想补充内容
            # 标记需要丢弃旧回复
            self.discard_responses.add(session_id)
            
            # 恢复上次发送的消息到缓冲区
            if session_id in self.last_sent_messages:
                buffer.messages = [self.last_sent_messages[session_id]]
            
            # 将新消息加入缓冲区
            buffer.add(message_text)
            
            if debug_mode:
                logger.info(f"[防抖] 检测到 LLM 回复前的新消息，已恢复上次消息并合并: {buffer.get_full_text()}")
            
            # 注意：不调用 stop_event()，让新消息继续正常的防抖流程
            # 继续往下执行，重新判断完整性
        elif session_id in self.waiting_sessions:
            # 已有其他消息在等待中，只添加到缓存
            buffer.add(message_text)
            event.stop_event()
            
            if debug_mode:
                logger.info(f"[防抖] 检测到正在等待中，添加到缓存: {buffer.get_full_text()}")
            return
        
        # 如果正在等待旧回复，先不添加新消息（已在上面添加过）
        if not (cancel_on_new and session_id in self.pending_llm_sessions):
            # 添加新消息到缓冲区
            buffer.add(message_text)
        full_text = buffer.get_full_text()
        
        # 模型预测
        score_send, _ = self.classifier.predict(full_text)
        is_complete = score_send >= threshold
        
        if debug_mode:
            logger.info(f"[防抖] 文本: '{full_text}' | 完整概率: {score_send:.4f} | 判定: {'发送' if is_complete else '等待'}")
        
        if is_complete:
            # 句子完整，记录要发送的内容并清空缓冲区
            self.last_sent_messages[session_id] = full_text
            buffer.clear()
            
            # 如果启用了取消功能，标记该会话正在等待 LLM 回复
            cancel_on_new = self._get_config("cancel_on_new_message", True)
            if cancel_on_new:
                self.pending_llm_sessions.add(session_id)
            
            # 修改请求内容为完整消息
            if hasattr(req, 'prompt') and req.prompt:
                req.prompt = full_text
            
            if debug_mode:
                logger.info(f"[防抖] 完整发送: {full_text}")
        else:
            # 句子未完整，进入等待循环
            if debug_mode:
                logger.info(f"[防抖] 句子未完整，进入等待: {full_text}")
            
            # 标记该会话正在等待
            self.waiting_sessions.add(session_id)
            initial_update_time = buffer.last_update
            
            try:
                # 循环等待直到超时或有新消息
                while True:
                    await asyncio.sleep(1)  # 每秒检查一次
                    
                    # 检查是否有新消息到来
                    if buffer.last_update > initial_update_time:
                        # 有新消息，重新判断完整性
                        full_text = buffer.get_full_text()
                        score_send, _ = self.classifier.predict(full_text)
                        is_complete = score_send >= threshold
                        
                        if debug_mode:
                            logger.info(f"[防抖] 检测到新消息，重新判断: '{full_text}' | 完整概率: {score_send:.4f} | 判定: {'发送' if is_complete else '等待'}")
                        
                        if is_complete:
                            # 现在完整了，发送
                            self.last_sent_messages[session_id] = full_text
                            buffer.clear()
                            
                            # 标记等待 LLM 回复
                            if cancel_on_new:
                                self.pending_llm_sessions.add(session_id)
                            
                            # 修改请求内容为完整消息
                            if hasattr(req, 'prompt') and req.prompt:
                                req.prompt = full_text
                            
                            if debug_mode:
                                logger.info(f"[防抖] 等待期间变完整，发送: {full_text}")
                            
                            return  # 不调用 stop_event()，允许请求继续
                        else:
                            # 还是不完整，更新时间，继续等待
                            initial_update_time = buffer.last_update
                    
                    # 检查是否超时
                    if buffer.is_timeout(timeout_seconds):
                        # 超时，发送当前缓存的内容
                        full_text = buffer.get_full_text()
                        buffer.clear()
                        
                        # 修改请求内容为缓存消息
                        if hasattr(req, 'prompt') and req.prompt:
                            req.prompt = full_text
                        
                        if debug_mode:
                            logger.info(f"[防抖] 等待超时，发送: {full_text}")
                        
                        return  # 不调用 stop_event()，允许请求继续
            
            finally:
                # 确保清除等待标记
                self.waiting_sessions.discard(session_id)
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        """LLM 回复后的钩子 - 检查是否需要丢弃回复"""
        session_id = event.message_obj.session_id
        debug_mode = self._get_config("debug_mode", False)
        
        # 检查是否需要丢弃这个回复
        if session_id in self.discard_responses:
            self.discard_responses.discard(session_id)
            self.pending_llm_sessions.discard(session_id)
            
            if debug_mode:
                logger.info(f"[防抖] 已丢弃过时的 LLM 回复")
            
            # 阻止回复发送 - 清空响应内容
            if hasattr(resp, 'completion_text'):
                resp.completion_text = ""
            return
        
        # 清除待处理标记
        self.pending_llm_sessions.discard(session_id)
        
        # 清除已发送消息的记录
        self.last_sent_messages.pop(session_id, None)
        
        if debug_mode:
            logger.info(f"[防抖] LLM 回复完成，清除会话 {session_id} 的待处理标记")
    
    async def _timeout_checker(self):
        """后台任务：定期清理过期的缓冲区"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查一次
            
            timeout_seconds = self._get_config("timeout_seconds", 30)
            if timeout_seconds <= 0:
                continue
            
            # 清理过期的缓冲区（超过3倍超时时间）
            expired_sessions = []
            for session_id, buffer in self.buffers.items():
                if buffer.messages and buffer.is_timeout(timeout_seconds * 3):
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
        self.pending_llm_sessions.clear()
        self.discard_responses.clear()
        self.last_sent_messages.clear()
        self.waiting_sessions.clear()
        self.classifier = None
        logger.info("🛑 消息防抖插件已卸载")
