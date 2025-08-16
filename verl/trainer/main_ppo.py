# Copyright 2024 Bytedance Ltd. ...
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class RewardManager():
    """Debug-friendly Reward Manager with mask-aware logging."""

    def __init__(self, tokenizer, num_examine,
                 structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score

    def __call__(self, data: DataProto):
        """Compute per-token rewards (mask-aware) with rich debug prints."""

        # 若已有 rm_scores，直接返回
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # ---- 优先读取 batch 级 sentence_rewards（若存在）----
        batch_sentence_rewards = None
        if hasattr(data, "non_tensor_batch") and isinstance(data.non_tensor_batch, dict):
            batch_sentence_rewards = data.non_tensor_batch.get('sentence_rewards', None)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            # ---------- 1) 计算有效 prompt / response 长度（优先使用 info_mask） ----------
            prompt_ids = data_item.batch['prompts']
            prompt_len = int(prompt_ids.shape[-1])

            info_mask = data_item.batch.get('info_mask', None)
            attn_mask = data_item.batch['attention_mask']

            if info_mask is not None:
                # 仅统计有效 prompt 位（与训练侧一致）
                valid_prompt_len = int(info_mask[:prompt_len].sum().item())
            else:
                valid_prompt_len = int(attn_mask[:prompt_len].sum().item())

            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            # responses（右侧）与其有效位置
            response_ids = data_item.batch['responses']

            if info_mask is not None:
                # 在“响应段”里，取出 info_mask == 1 的位置，作为“响应-only”的映射索引
                info_mask_resp = info_mask[prompt_len:]
                response_positions = torch.nonzero(info_mask_resp, as_tuple=False).squeeze(-1)
                valid_response_len = int(response_positions.numel())
                # 仅包含“生成 token”的响应视图（连续坐标系）
                valid_response_ids = response_ids[response_positions]
            else:
                # 兼容无 info_mask 的情况：用 attention_mask 切分
                attn_mask_resp = attn_mask[prompt_len:]
                valid_response_len = int(attn_mask_resp.sum().item())
                response_positions = torch.arange(valid_response_len, device=response_ids.device)
                valid_response_ids = response_ids[:valid_response_len]

            # ---------- 2) 可读性解码 ----------
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=True)
            full_resp_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # ---------- 3) 读取句末奖励（batch 级优先） ----------
            if batch_sentence_rewards is not None:
                rewards = batch_sentence_rewards[i]
            else:
                rewards = data_item.non_tensor_batch.get('sentence_rewards', [])

            # ---------- 4) 打印与写入 ----------
            print(f"[DEBUG] sample={i} valid_response_len={valid_response_len}  "
                  f"(mask={'info_mask' if info_mask is not None else 'attention_mask'})")

            for pos, val in list(rewards):
                # 逻辑位置与数值规范化
                try:
                    pos = int(pos)
                    val = float(val)
                except Exception:
                    print(f"[WARN] sample={i} skip invalid reward item: {(pos, val)}")
                    continue

                # 片段句子：回溯最多 32 个 token（在“响应-only”视图里）
                frag_start = max(0, pos - 32)
                frag_end = min(valid_response_len, pos + 1)
                frag_tokens = valid_response_ids[frag_start:frag_end]
                frag_text = self.tokenizer.decode(frag_tokens, skip_special_tokens=True)

                # 句末 token piece（响应-only 视图）
                if 0 <= pos < valid_response_len:
                    end_tok_id = int(valid_response_ids[pos].item())
                    end_piece_list = self.tokenizer.convert_ids_to_tokens([end_tok_id])
                    end_piece = end_piece_list[0] if end_piece_list else "<UNK>"
                    # 映射回 responses 的真实索引（写入用）
                    reward_pos = int(response_positions[pos].item())
                else:
                    end_piece = "<OUT_OF_RANGE>"
                    reward_pos = None

                # —— 无条件打印 ——（模仿示例风格）
                print(
                    f" 响应完整内容：{full_resp_text}；"
                    f"句子（可以是片段句子）：{frag_text}；"
                    f"句末token：{end_piece}；   "
                    f"句末token位置（响应-only）：{pos}；"
                    f"映射到responses的真实索引：{reward_pos if reward_pos is not None else 'N/A'}；"
                    f"句末奖励：{val}"
                )

                # 合法时写入：注意写到“真实索引” reward_pos
                if reward_pos is not None:
                    reward_tensor[i, reward_pos] = val

            # 可选：每条样本打印一次完整序列（含 prompt）
            print(sequences_str)

            # （保留原先的数据源抽样打印逻辑）
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            already_print_data_sources[data_source] = already_print_data_sources.get(data_source, 0)
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # 也能在这里再次打印 sequences_str（若想区分来源）

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
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

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
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
