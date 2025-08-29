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
        sample_confusions = []

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

            response_str = self.tokenizer.decode(response_ids[response_positions])
            all_sentence_count = len([s for s in re.split(r'[.!?]', response_str) if s.strip()])

            rewards = data_item.non_tensor_batch.get('sentence_rewards', [])
            tp = fp = tn = fn = 0
            for pos, val in rewards:
                if pos < valid_response_length:
                    reward_pos = int(response_positions[pos].item())
                    reward_tensor[i, reward_pos] = val
                    if val == 2:
                        tp += 1
                    elif val == -2:
                        fn += 1
                    elif val == -1:
                        fp += 1
                    elif val == 1:
                        tn += 1
                    #end_token_id = response_ids[reward_pos].item()
                    #end_token_str = self.tokenizer.decode([end_token_id])
                    #start_slice = max(0, pos - 20)
                    #snippet_ids = response_ids[response_positions[start_slice:pos + 1]].tolist()
                    #snippet_str = self.tokenizer.decode(snippet_ids)
                    #print( f"句末token: {end_token_str}, 句末token索引: {reward_pos}, 句子片段: {snippet_str}，句末奖励：{val}" )

            sample_confusions.append({
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'decisions': tp + fp + tn + fn,
                'sentences': all_sentence_count,
            })

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

        # Compute decision-level confusion matrix statistics
        if sample_confusions:
            def _pct(num, denom):
                return num / denom if denom > 0 else 0.0

            tp_micro = sum(s['tp'] for s in sample_confusions)
            fp_micro = sum(s['fp'] for s in sample_confusions)
            tn_micro = sum(s['tn'] for s in sample_confusions)
            fn_micro = sum(s['fn'] for s in sample_confusions)
            dec_micro = sum(s['decisions'] for s in sample_confusions)
            sent_micro = sum(s['sentences'] for s in sample_confusions)

            tp_pct_micro = _pct(tp_micro, dec_micro)
            fn_pct_micro = _pct(fn_micro, dec_micro)
            fp_pct_micro = _pct(fp_micro, dec_micro)
            tn_pct_micro = _pct(tn_micro, dec_micro)

            tp_pct_micro_all = _pct(tp_micro, sent_micro)
            fn_pct_micro_all = _pct(fn_micro, sent_micro)
            fp_pct_micro_all = _pct(fp_micro, sent_micro)
            tn_pct_micro_all = _pct(tn_micro, sent_micro)

            prec_micro = _pct(tp_micro, tp_micro + fp_micro)
            rec_micro = _pct(tp_micro, tp_micro + fn_micro)
            f1_micro = (
                2 * prec_micro * rec_micro / (prec_micro + rec_micro)
                if (prec_micro + rec_micro) > 0
                else 0.0
            )
            cov_micro = _pct(dec_micro, sent_micro)

            tp_dec_list = []
            fn_dec_list = []
            fp_dec_list = []
            tn_dec_list = []
            tp_all_list = []
            fn_all_list = []
            fp_all_list = []
            tn_all_list = []
            prec_list = []
            rec_list = []
            f1_list = []
            cov_list = []
            for s in sample_confusions:
                dec = s['decisions']
                sent = s['sentences']
                tp, fp, tn, fn = s['tp'], s['fp'], s['tn'], s['fn']
                tp_dec_list.append(_pct(tp, dec))
                fn_dec_list.append(_pct(fn, dec))
                fp_dec_list.append(_pct(fp, dec))
                tn_dec_list.append(_pct(tn, dec))
                tp_all_list.append(_pct(tp, sent))
                fn_all_list.append(_pct(fn, sent))
                fp_all_list.append(_pct(fp, sent))
                tn_all_list.append(_pct(tn, sent))
                prec = _pct(tp, tp + fp)
                rec = _pct(tp, tp + fn)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                prec_list.append(prec)
                rec_list.append(rec)
                f1_list.append(f1)
                cov_list.append(_pct(dec, sent))

            tp_pct_macro = np.mean(tp_dec_list) if tp_dec_list else 0.0
            fn_pct_macro = np.mean(fn_dec_list) if fn_dec_list else 0.0
            fp_pct_macro = np.mean(fp_dec_list) if fp_dec_list else 0.0
            tn_pct_macro = np.mean(tn_dec_list) if tn_dec_list else 0.0
            tp_pct_macro_all = np.mean(tp_all_list) if tp_all_list else 0.0
            fn_pct_macro_all = np.mean(fn_all_list) if fn_all_list else 0.0
            fp_pct_macro_all = np.mean(fp_all_list) if fp_all_list else 0.0
            tn_pct_macro_all = np.mean(tn_all_list) if tn_all_list else 0.0
            prec_macro = np.mean(prec_list) if prec_list else 0.0
            rec_macro = np.mean(rec_list) if rec_list else 0.0
            f1_macro = np.mean(f1_list) if f1_list else 0.0
            cov_macro = np.mean(cov_list) if cov_list else 0.0

            print(
                f"[DEBUG][Decision][Micro] per_decision TP%: {tp_pct_micro:.4f}, FN%: {fn_pct_micro:.4f}, FP%: {fp_pct_micro:.4f}, TN%: {tn_pct_micro:.4f}, "
                f"Precision: {prec_micro:.4f}, Recall: {rec_micro:.4f}, F1: {f1_micro:.4f}"
            )
            print(
                f"[DEBUG][Decision][Micro] per_all     TP%: {tp_pct_micro_all:.4f}, FN%: {fn_pct_micro_all:.4f}, FP%: {fp_pct_micro_all:.4f}, TN%: {tn_pct_micro_all:.4f}, "
                f"DecisionCoverage: {cov_micro:.4f}"
            )
            print(
                f"[DEBUG][Decision][Macro] per_decision TP%: {tp_pct_macro:.4f}, FN%: {fn_pct_macro:.4f}, FP%: {fp_pct_macro:.4f}, TN%: {tn_pct_macro:.4f}, "
                f"Precision: {prec_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}"
            )
            print(
                f"[DEBUG][Decision][Macro] per_all     TP%: {tp_pct_macro_all:.4f}, FN%: {fn_pct_macro_all:.4f}, FP%: {fp_pct_macro_all:.4f}, TN%: {tn_pct_macro_all:.4f}, "
                f"DecisionCoverage: {cov_macro:.4f}"
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
