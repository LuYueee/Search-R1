# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # ---- 优先读取 batch 级 sentence_rewards（与 generation v49 的传递方式对齐）----
        batch_sentence_rewards = None
        if hasattr(data, "non_tensor_batch") and isinstance(data.non_tensor_batch, dict):
            batch_sentence_rewards = data.non_tensor_batch.get('sentence_rewards', None)
        # -------------------------------------------------------------------------
            
        for i in range(len(data)):
            data_item = data[i]

            # 取 prompt/response 的有效长度（显式转为 Python int）
            prompt_ids = data_item.batch['prompts']
            prompt_length = int(prompt_ids.shape[-1])
            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # 解码：完整序列（含 prompt）和仅响应文本
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            full_resp_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # —— 拿句末奖励（先 batch 级，再回退 item 级）——
            if batch_sentence_rewards is not None:
                rewards = batch_sentence_rewards[i]
            else:
                rewards = data_item.non_tensor_batch.get('sentence_rewards', [])
            # -----------------------------------------
            #print("[DEBUG]reward:",rewards)
            # —— 打印 & 写入：每个句末 (pos, val) —— 
            for pos, val in list(rewards):
                pos = int(pos)
                val = float(val)

                # 片段句子：从句末回溯最多 32 个 response token
                frag_start = max(0, pos - 32)
                frag_end = min(valid_response_length, pos + 1)  # 保护上界
                frag_tokens = response_ids[frag_start:frag_end]
                frag_text = self.tokenizer.decode(frag_tokens, skip_special_tokens=True)

                # 句末 token 的 piece（越界时给提示）
                if 0 <= pos < valid_response_length:
                    end_tok_id = int(response_ids[pos].item())
                    end_piece_list = self.tokenizer.convert_ids_to_tokens([end_tok_id])
                    end_piece = end_piece_list[0] if end_piece_list else "<UNK>"
                else:
                    end_piece = "<OUT_OF_RANGE>"

                # —— 无条件打印 —— 
                print(
                    f" 响应完整内容：{full_resp_text}；"
                    f"句子（可以是片段句子）：{frag_text}；"
                    f"句末token：{end_piece}；   "
                    f"句末token位置：{pos}；"
                    f"句末奖励：{val}"
                )

                # 仅在合法索引时写入 step-level 奖励（对齐 response 段）
                if 0 <= pos < valid_response_length:
                    reward_tensor[i, pos] = val

            # 保留原有的每样本完整序列打印（含 prompt，可按需注释）
            print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              structure_format_score=config.reward_model.structure_format_score, 
                              final_format_score=config.reward_model.final_format_score,
                              retrieval_score=config.reward_model.retrieval_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
