import threading
import logging
from pathlib import Path
from typing import Any, Optional, Tuple
from PIL import Image
import io
import time
import base64
import requests
import json
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import pack, unpack, repeat, reduce, rearrange
from utils import extract_state_dict
from utils import init_weights, LossWithIntermediateLosses

from prompts.prompt_task import prompt_task
from prompts.task2actions import task2actions
from prompts.prompt_action_will_do import prompt_action_will_do

class Agent(nn.Module):
    def __init__(self, model,cfg):
        """强化学习智能体，集成环境交互、动作决策、模型训练等功能
        
        参数:
            model: 核心决策模型(如Actor-Critic网络)
            cfg: 配置字典，包含训练参数等
        """
        super().__init__()
        self.loss_weight = cfg['training']['agent']['loss_weight'] #各损失项的权重
        self.model = model # 核心神经网络模型
        self.batch = {} # 存储当前批次的数据
        self._running = True # 运行状态标志
        self.buffer_lock = threading.Lock() # 线程锁，用于多线程数据同步

        # 初始化日志系统
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO) 
        
        # 加载CLIP图像预处理模型
        import open_clip
        _, _, self.image_preprocess = open_clip.create_model_and_transforms(cfg['image_process']['clip_arch'], pretrained=cfg['image_process']['clip_path'])
    
    
    @property
    def device(self):
        """获取模型所在设备(CPU/GPU)"""
        return self.actor_critic.conv1.weight.device # 通过参数位置推断设备


    def __repr__(self) -> str:
        return "agent"
    
    
    def reset(self):
        self._running = True
        self.batch = {} # 清空批次数据
        
        
    def load(self, path_to_checkpoint: Path, device: torch.device) -> None:
        """加载预训练权重
        
        参数:
            path_to_checkpoint: 模型权重文件路径
            device: 加载设备（CPU/GPU）
        """
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        new_state_dict = OrderedDict() # 调整键名以匹配当前模型结构（移除部分前缀）
        for k, v in agent_state_dict.items():
            name = k[:6]+k[13:]  # 用于删掉模型中的层的weight字符串
            new_state_dict[name] = v
        agent_state_dict = new_state_dict
        self.load_state_dict(agent_state_dict)


    def act(self, batch: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        """生成动作（推理阶段使用）
        
        参数:
            batch: 输入数据批次，包含观测、状态、指令
            should_sample: 是否采样动作（否则取argmax）
            temperature: 采样温度参数，控制随机性
        返回:
            动作张量
        """
        
        with torch.no_grad(): # 禁用梯度计算
            obs = batch['observations']
            states = batch['states']
            texts = batch['instr']
        # 调用模型前向传播获取动作预测
        predict_actions, prediction = self(obs,texts,states)
        return predict_actions


    def update_target_tokenizer(self):
        '''
        动量更新 target_tokenizer 的参数
        '''
        if self.tokenizer is None:
            return
        for model_param, shadow_param in zip(self.tokenizer.parameters(), self.target_tokenizer.parameters()):
            shadow_param.data = (1.0 - self.momentum) * shadow_param.data + self.momentum * model_param.data
        self.momentum = min(1., self.momentum+self.momentum_delta)


    def forward(self,obs,texts,states):
        """前向传播（训练阶段使用）
        
        返回:
            actions: 预测动作
            prediction: 可能的下状态预测（如自监督目标）
        """
        actions,prediction = self.model(obs,texts,states)
        return actions, prediction
    
    
    def compute_loss(self, batch, **kwargs: Any) -> LossWithIntermediateLosses:
        """计算总损失
        
        包含:
            1. 动作预测交叉熵损失
            2. 状态预测MSE损失（自监督）
        其中真实的observation经过self.model的编码处理,得到一个token,与预测的token进行比较
        """
        # 解包批次数据
        obs = batch['observations']
        next_obs = batch['next_observations']
        states = batch['states']
        texts = batch['instr']
        actions = batch['actions']
        
        # 前向传播
        predict_actions, prediction = self(obs,texts,states) 
        
        # 动作损失计算（交叉熵）
        loss_actions = (F.cross_entropy(predict_actions.reshape(-1,256).float(), actions.reshape(-1,).long()) ) * self.loss_weight[0]
        
        loss_observation=loss_actions*0

        if prediction is not None: # 如果模型有返回observation的话
            # 临时切换为评估模式以计算目标嵌入
            training=self.training
            self.eval()
            with torch.no_grad():
                next_token,text_token = self.model(next_obs,texts,states,return_embed=True)
            loss_observation = F.mse_loss(prediction, next_token)*self.loss_weight[1]
            
            if training: # 恢复原始训练模式
                self.train()
        return LossWithIntermediateLosses(loss_actions=loss_actions,loss_observation=loss_observation)


    def update_input(self,batch):
        '''
        更新batch数据, 首先将batch数据拷贝一份, 然后将拷贝的数据存储到self.batch中
        备份的数据用于后续的推理, self.batch用于训练
        '''
        with self.buffer_lock:
            for k in batch.keys():
                # 深拷贝数据避免被外部修改
                if isinstance(batch[k],torch.Tensor):
                    self.batch[k] = batch[k].clone().detach()
                elif isinstance(batch[k],str):
                        self.batch[k] = batch[k]
                else:
                    self.batch[k] = batch[k].copy()
    
    def update_instr(self):
        """持续更新指令（独立线程运行）
        
        通过HTTP请求与语言模型交互，动态生成动作指令
        """
        while self.batch is None or self.batch =={}: # 等待batch数据
            time.sleep(0.1)
        with self.buffer_lock: # 加锁
            batch = {}
            for k in self.batch.keys():
                if isinstance(self.batch[k],torch.Tensor):
                    batch[k] = self.batch[k].clone().detach()
                elif isinstance(self.batch[k],str):
                    batch[k] = self.batch[k]
                else:
                    batch[k] = self.batch[k].copy()

        url = 'http://127.0.0.1:8000/inference/'
        
        def pil_image_to_base64(image):  # 将图片转换为base64编码
            buffered = io.BytesIO()
            image.save(buffered, format="PNG") 
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        def send_request(image,prompt): # 发送请求
            data = {
                'image': pil_image_to_base64(image),  
                'prompt': prompt
            }

            response = requests.post(url, json=data)

            if response.status_code == 200: # 如果返回状态码为200
                answer=response.json()['response'][3:-4].strip() # 获取回答
                return answer 
            else:
                assert False, f"Failed to get valid response, status code: {response.status_code}"  # 如果返回状态码不为200，抛出异常
                self.logger.debug(f"Failed to get valid response, status code: {response.status_code}")

        while True:
            try:
                image = batch['mat']
                task = batch['instr'][0]
                prompt = prompt_task.format(task=task)
                answer = send_request(image, prompt)
                answer = json.loads(answer)
                task_format = answer['task']
                actions = task2actions[task_format]
                break
            except Exception as e:
                self.logger.debug(f'error {e}')

        while self._running:
            try:
                with self.buffer_lock:
                    batch = {}
                    for k in self.batch.keys():
                        if isinstance(self.batch[k],torch.Tensor):
                            batch[k] = self.batch[k].clone().detach()
                        elif isinstance(self.batch[k],str):
                            batch[k] = self.batch[k]
                        else:
                            batch[k] = self.batch[k].copy()
                image = batch['mat']
                task = batch['instr'][0]
                prompt = prompt_action_will_do.format(task=task,actions=actions,hand_state=batch['hand_state'])
                answer = send_request(image, prompt)
                answer = answer.replace('\_', '_')
                answer = json.loads(answer)
                action = answer['action']
                with self.buffer_lock:
                    self.batch['new_instr'] = [task+f'; action: {action}']
                self.logger.debug(f'action {action}')
                time.sleep(0.1)
            except Exception as e:
                self.logger.debug(f'error {e}')


    def update_output(self):
        '''
        更新输出，当batch中有新的instr时，调用act函数进行推理
        '''
        self.logger.debug('start update_output')
        while self.batch is None or self.batch =={} or 'new_instr' not in self.batch:
            time.sleep(0.01)
        with self.buffer_lock:
            batch = {}
            for k in self.batch.keys():
                if isinstance(self.batch[k],torch.Tensor):
                    batch[k] = self.batch[k].clone().detach()
                elif isinstance(self.batch[k],str):
                    batch[k] = self.batch[k]
                else:
                    batch[k] = self.batch[k].copy()
        self.logger.debug(f"new_instr {batch['new_instr']}")
        batch['instr'] = batch['new_instr']
        action = self.act(batch)
        return action
