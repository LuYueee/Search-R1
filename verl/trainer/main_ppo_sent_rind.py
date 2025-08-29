# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
        return qa_em_format.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0., lambda_episode=1.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score
        self.lambda_episode = lambda_episode

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        all_scores = []
        turns_stats = data.meta_info.get('turns_stats', [])
        confusion_stats = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            info_mask = data_item.batch.get('info_mask')

            if info_mask is not None:
                valid_prompt_length = int(info_mask[:prompt_length].sum().item())
            else:
                valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            if info_mask is not None:
                info_mask_resp = info_mask[prompt_length:]
                response_positions = torch.nonzero(info_mask_resp, as_tuple=False).squeeze(-1)
                valid_response_length = response_positions.shape[0]
            else:
                attention_mask_resp = data_item.batch['attention_mask'][prompt_length:]
                valid_response_length = int(attention_mask_resp.sum().item())
                response_positions = torch.arange(valid_response_length, device=response_ids.device)

            sequences = torch.cat((valid_prompt_ids, response_ids[response_positions]))
            sequences_str = self.tokenizer.decode(sequences)

            rewards = data_item.non_tensor_batch.get('sentence_rewards', [])
            sample_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
            for pos, val in rewards:
                if pos < valid_response_length:
                    reward_pos = int(response_positions[pos].item())
                    reward_tensor[i, reward_pos] = val
                    label = {2: 'TP', -2: 'FN', -1: 'FP', 1: 'TN'}.get(int(val))
                    if label is not None:
                        sample_counts[label] += 1
                    #end_token_id = response_ids[reward_pos].item()
                    #end_token_str = self.tokenizer.decode([end_token_id])
                    #start_slice = max(0, pos - 20)
                    #snippet_ids = response_ids[response_positions[start_slice:pos + 1]].tolist()
                    #snippet_str = self.tokenizer.decode(snippet_ids)
                    #print( f"句末token: {end_token_str}, 句末token索引: {reward_pos}, 句子片段: {snippet_str}，句末奖励：{val}" )
            decision_points = sum(sample_counts.values())
            total_sents = turns_stats[i] if i < len(turns_stats) else decision_points
            confusion_stats.append((sample_counts, decision_points, total_sents))

            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                structure_format_score=self.structure_format_score,
                final_format_score=self.final_format_score,
                retrieval_score=self.retrieval_score,
                format_score=self.format_score,
            )
            if valid_response_length > 0:
                last_pos = int(response_positions[valid_response_length - 1].item())
                reward_tensor[i, last_pos] = reward_tensor[i, last_pos] + self.lambda_episode * score
                all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if all_scores:
            print(
                f"[DEBUG][EM+Format] Batch reward stats - mean: {np.mean(all_scores):.4f}, "
                f"min: {np.min(all_scores):.4f}, max: {np.max(all_scores):.4f}"
            )

        # Compute step-level statistics across all valid response tokens
        prompt_len = data.batch['prompts'].shape[1]
        if 'info_mask' in data.batch:
            resp_mask = data.batch['info_mask'][:, prompt_len:]
        else:
            resp_mask = data.batch['attention_mask'][:, prompt_len:]
        resp_mask = resp_mask.to(reward_tensor.device)
        valid_rewards = reward_tensor[resp_mask.bool()]

        if valid_rewards.numel() > 0:
            vr = valid_rewards.cpu().numpy()
            step_mean = vr.mean()
            step_std = vr.std()
            step_min = vr.min()
            step_max = vr.max()
            pos_rate = (vr > 0).mean()
            neg_rate = (vr < 0).mean()
        else:
            step_mean = step_std = step_min = step_max = pos_rate = neg_rate = 0.0

        print(
            f"[DEBUG][Step] step_reward/mean: {step_mean:.4f}, /std: {step_std:.4f}, /min: {step_min:.4f}, "
            f"/max: {step_max:.4f}, /pos_rate: {pos_rate:.4f}, /neg_rate: {neg_rate:.4f}"
        )

        # Compute per-sample aggregated rewards within batch
        sample_mask = resp_mask.bool()
        sample_returns = (reward_tensor * sample_mask).sum(dim=1)
        nonzero_steps = (reward_tensor * sample_mask != 0).sum(dim=1)
        sample_lengths = sample_mask.sum(dim=1)

        sr = sample_returns.cpu().numpy()
        if sr.size > 0:
            return_mean = sr.mean()
            return_std = sr.std()
            return_min = sr.min()
            return_max = sr.max()
            nz_steps_mean = nonzero_steps.float().mean().item()
            len_mean = sample_lengths.float().mean().item()
        else:
            return_mean = return_std = return_min = return_max = 0.0
            nz_steps_mean = len_mean = 0.0

        print(
            f"[DEBUG][Step] sample/return_mean: {return_mean:.4f}, /std: {return_std:.4f}, /min: {return_min:.4f}, "
            f"/max: {return_max:.4f}, /nonzero_steps_mean: {nz_steps_mean:.4f}, /len_mean: {len_mean:.4f}"
        )

        # Compute confusion matrix statistics for gating decisions
        micro_counts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        micro_decisions = 0
        micro_total_sents = 0
        macro_dec_ratios = {k: [] for k in micro_counts}
        macro_all_ratios = {k: [] for k in micro_counts}
        macro_precisions, macro_recalls, macro_f1s = [], [], []
        macro_coverages = []

        for counts, dec_points, total_sents in confusion_stats:
            micro_decisions += dec_points
            micro_total_sents += total_sents
            for k in micro_counts:
                micro_counts[k] += counts[k]

            if dec_points > 0:
                for k in counts:
                    macro_dec_ratios[k].append(counts[k] / dec_points)
                precision = counts['TP'] / (counts['TP'] + counts['FP']) if (counts['TP'] + counts['FP']) > 0 else 0.0
                recall = counts['TP'] / (counts['TP'] + counts['FN']) if (counts['TP'] + counts['FN']) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                for k in counts:
                    macro_dec_ratios[k].append(0.0)
                precision = recall = f1 = 0.0
            macro_precisions.append(precision)
            macro_recalls.append(recall)
            macro_f1s.append(f1)

            if total_sents > 0:
                for k in counts:
                    macro_all_ratios[k].append(counts[k] / total_sents)
                macro_coverages.append(dec_points / total_sents)
            else:
                for k in counts:
                    macro_all_ratios[k].append(0.0)
                macro_coverages.append(0.0)

        if micro_decisions > 0:
            micro_dec_pct = {k: micro_counts[k] / micro_decisions for k in micro_counts}
        else:
            micro_dec_pct = {k: 0.0 for k in micro_counts}
        if micro_total_sents > 0:
            micro_all_pct = {k: micro_counts[k] / micro_total_sents for k in micro_counts}
            micro_coverage = micro_decisions / micro_total_sents
        else:
            micro_all_pct = {k: 0.0 for k in micro_counts}
            micro_coverage = 0.0
        micro_precision = micro_counts['TP'] / (micro_counts['TP'] + micro_counts['FP']) if (micro_counts['TP'] + micro_counts['FP']) > 0 else 0.0
        micro_recall = micro_counts['TP'] / (micro_counts['TP'] + micro_counts['FN']) if (micro_counts['TP'] + micro_counts['FN']) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        macro_dec_pct = {k: float(np.mean(v)) if v else 0.0 for k, v in macro_dec_ratios.items()}
        macro_all_pct = {k: float(np.mean(v)) if v else 0.0 for k, v in macro_all_ratios.items()}
        macro_precision = float(np.mean(macro_precisions)) if macro_precisions else 0.0
        macro_recall = float(np.mean(macro_recalls)) if macro_recalls else 0.0
        macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0
        macro_coverage = float(np.mean(macro_coverages)) if macro_coverages else 0.0

        print(
            f"[DEBUG][Confusion] micro(decision)/TP%: {micro_dec_pct['TP']:.4f}, /FN%: {micro_dec_pct['FN']:.4f}, "
            f"/FP%: {micro_dec_pct['FP']:.4f}, /TN%: {micro_dec_pct['TN']:.4f}, /Precision: {micro_precision:.4f}, "
            f"/Recall: {micro_recall:.4f}, /F1: {micro_f1:.4f}"
        )
        print(
            f"[DEBUG][Confusion] macro(decision)/TP%: {macro_dec_pct['TP']:.4f}, /FN%: {macro_dec_pct['FN']:.4f}, "
            f"/FP%: {macro_dec_pct['FP']:.4f}, /TN%: {macro_dec_pct['TN']:.4f}, /Precision: {macro_precision:.4f}, "
            f"/Recall: {macro_recall:.4f}, /F1: {macro_f1:.4f}"
        )
        print(
            f"[DEBUG][Confusion] micro(all)/TP%: {micro_all_pct['TP']:.4f}, /FN%: {micro_all_pct['FN']:.4f}, "
            f"/FP%: {micro_all_pct['FP']:.4f}, /TN%: {micro_all_pct['TN']:.4f}, /Precision: {micro_precision:.4f}, "
            f"/Recall: {micro_recall:.4f}, /F1: {micro_f1:.4f}, /DecisionCoverage: {micro_coverage:.4f}"
        )
        print(
            f"[DEBUG][Confusion] macro(all)/TP%: {macro_all_pct['TP']:.4f}, /FN%: {macro_all_pct['FN']:.4f}, "
            f"/FP%: {macro_all_pct['FP']:.4f}, /TN%: {macro_all_pct['TN']:.4f}, /Precision: {macro_precision:.4f}, "
            f"/Recall: {macro_recall:.4f}, /F1: {macro_f1:.4f}, /DecisionCoverage: {macro_coverage:.4f}"
        )

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
                              retrieval_score=config.reward_model.retrieval_score,
                              lambda_episode=config.reward_model.lambda_episode)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1,
    structure_format_score=config.reward_model.structure_format_score, 
    final_format_score=config.reward_model.final_format_score,
    retrieval_score=config.reward_model.retrieval_score,
    lambda_episode=config.reward_model.lambda_episode)

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
