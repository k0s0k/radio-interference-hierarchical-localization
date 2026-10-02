"""Behaviour-cloning warm start followed by clipped PPO; all training is offline.

PPO objective follows Schulman et al. (2017). Gamma=1 targets undiscounted total
macro-action time. GAE lambda controls variance, not the task's time objective.
Training, imitation, validation and final test use disjoint scenario seed banks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .environment import SearchEnvironment, heuristic_index
from .policy import CandidateActorCritic, tensor_observations, save_policy


@torch.no_grad()
def validate(model, n=16, first_seed=30000):
    times, successes = [], []
    env = SearchEnvironment()
    for seed in range(first_seed, first_seed + n):
        observation = env.reset(seed)
        while not env.done:
            action, _ = model.choose(observation)
            observation, _, _, _ = env.step(action)
        outcome = env.sim.outcome()
        times.append(outcome["virtual_s"])
        successes.append(outcome["all_clear"] and env.success_certificate)
    return dict(mean_s=float(np.mean(times)), all_clear=float(np.mean(successes)), n=n)


def imitation_data(n_episodes):
    env = SearchEnvironment()
    observations, labels, cost_returns = [], [], []
    for seed in range(20000, 20000 + n_episodes):
        observation = env.reset(seed)
        episode_rewards = []
        while not env.done:
            index = heuristic_index(env.actions)
            observations.append(observation)
            labels.append(index)
            observation, reward, _, _ = env.step(index)
            episode_rewards.append(reward)
        if not env.success_certificate:
            raise RuntimeError("Teacher failed: investigate before training")
        cost_returns.extend(np.cumsum(episode_rewards[::-1])[::-1].copy().tolist())
    return (
        tensor_observations(observations),
        torch.tensor(labels),
        torch.tensor(cost_returns, dtype=torch.float32),
    )


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = CandidateActorCritic(args.hidden)
    started = time.perf_counter()
    config = vars(args).copy()
    config.update(
        seed_banks=dict(
            train=[10000, 19999],
            imitation=[20000, 20000 + args.teacher_episodes - 1],
            validation=[30000, 30000 + args.validation - 1],
            test=[40000, 49999],
        ),
        architecture="shared candidate MLP + masked pooling; no attention/GRU",
        gamma=1.0,
        gae_lambda=0.95,
        clip=0.2,
        reward="-macro_virtual_seconds/1000; -100 on failed episode",
        python_torch=str(torch.__version__),
    )
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Collecting geometry-teacher demonstrations...", flush=True)
    demonstrations, labels, teacher_returns = imitation_data(args.teacher_episodes)
    bc_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.bc_epochs):
        losses, correct = [], 0
        for indices in torch.randperm(len(labels)).split(256):
            observation = {k: v[indices] for k, v in demonstrations.items()}
            distribution, predicted_returns = model(observation)
            loss = -distribution.log_prob(labels[indices]).mean()
            loss = (
                loss
                + 0.05 * (predicted_returns - teacher_returns[indices]).square().mean()
            )
            bc_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            bc_optimizer.step()
            losses.append(float(loss.detach()))
            correct += int((distribution.logits.argmax(-1) == labels[indices]).sum())
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"BC epoch={epoch+1} loss={np.mean(losses):.4f} accuracy={correct/len(labels):.3f}",
                flush=True,
            )
    warm_validation = validate(model, args.validation)
    save_policy(
        output / "bc.pt",
        model,
        dict(stage="behaviour_cloning", validation=warm_validation),
    )
    print("BC validation:", warm_validation, flush=True)
    # Keep BC separately. best_ppo.pt must actually have undergone PPO updates.
    best_score = float("inf")
    best_validation = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    envs = [SearchEnvironment() for _ in range(args.envs)]
    observations = [env.reset(int(rng.integers(10000, 20000))) for env in envs]
    episode_costs = [0.0] * args.envs
    recent_returns, recent_successes = [], []
    total_steps = 0
    rollout_n = args.rollout_steps
    updates = math_ceil(args.steps / (rollout_n * args.envs))
    history = []
    for update in range(1, updates + 1):
        batch_obs, batch_actions, batch_logs, batch_values = [], [], [], []
        rewards, dones = [], []
        for _ in range(rollout_n):
            obs_t = tensor_observations(observations)
            with torch.no_grad():
                distribution, values = model(obs_t)
                actions = distribution.sample()
                logs = distribution.log_prob(actions)
            batch_obs.append(obs_t)
            batch_actions.append(actions)
            batch_logs.append(logs)
            batch_values.append(values)
            step_rewards, step_dones = [], []
            for i, env in enumerate(envs):
                observations[i], reward, done, info = env.step(int(actions[i]))
                step_rewards.append(reward)
                step_dones.append(done)
                episode_costs[i] += reward
                if done:
                    recent_returns.append(episode_costs[i])
                    recent_successes.append(info["certificate"])
                    episode_costs[i] = 0.0
                    observations[i] = env.reset(int(rng.integers(10000, 20000)))
            rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
            dones.append(torch.tensor(step_dones, dtype=torch.float32))
            total_steps += args.envs
        with torch.no_grad():
            _, next_values = model(tensor_observations(observations))
        values = torch.stack(batch_values)
        rewards_t, dones_t = torch.stack(rewards), torch.stack(dones)
        advantages = torch.zeros_like(rewards_t)
        gae = torch.zeros(args.envs)
        for t in reversed(range(rollout_n)):
            next_value = next_values if t == rollout_n - 1 else values[t + 1]
            nonterminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + next_value * nonterminal - values[t]
            gae = delta + 0.95 * nonterminal * gae
            advantages[t] = gae
        returns = (advantages + values).flatten()
        flat_advantages = advantages.flatten()
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )
        flat_obs = {k: torch.cat([o[k] for o in batch_obs]) for k in batch_obs[0]}
        flat_actions, old_logs = torch.cat(batch_actions), torch.cat(batch_logs)
        losses, entropies, kls, clip_fractions = [], [], [], []
        for epoch in range(args.epochs):
            stop_epoch = False
            for indices in torch.randperm(len(flat_actions)).split(args.batch_size):
                distribution, predictions = model(
                    {k: v[indices] for k, v in flat_obs.items()}
                )
                new_logs = distribution.log_prob(flat_actions[indices])
                log_ratio = new_logs - old_logs[indices]
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(
                    ratio * flat_advantages[indices],
                    ratio.clamp(0.8, 1.2) * flat_advantages[indices],
                ).mean()
                value_loss = 0.5 * (predictions - returns[indices]).square().mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - args.entropy * entropy
                with torch.no_grad():
                    kl = float(((ratio - 1) - log_ratio).mean())
                if kl > 0.05:
                    stop_epoch = True
                    break
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach()))
                entropies.append(float(entropy.detach()))
                kls.append(kl)
                clip_fractions.append(
                    float(((ratio - 1).abs() > 0.2).float().mean().detach())
                )
            if stop_epoch:
                break
        row = dict(
            update=update,
            steps=total_steps,
            train_return=(
                float(np.mean(recent_returns[-40:])) if recent_returns else 0.0
            ),
            train_success=(
                float(np.mean(recent_successes[-40:])) if recent_successes else 0.0
            ),
            loss=float(np.mean(losses)) if losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            approximate_kl=float(np.mean(kls)) if kls else 0.0,
            clip_fraction=float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            elapsed_s=time.perf_counter() - started,
            validation_mean_s="",
            validation_success="",
        )
        if update == 1 or update % args.validate_every == 0 or update == updates:
            result = validate(model, args.validation)
            score = result["mean_s"] + (1 - result["all_clear"]) * 1e6
            row.update(
                validation_mean_s=result["mean_s"],
                validation_success=result["all_clear"],
            )
            if score < best_score:
                best_score, best_validation = score, result
                save_policy(
                    output / "best_ppo.pt",
                    model,
                    dict(
                        stage="bc_then_ppo",
                        steps=total_steps,
                        validation=result,
                        training_seed=args.seed,
                    ),
                )
        history.append(row)
        with (output / "training.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
    save_policy(
        output / "last_ppo.pt", model, dict(stage="bc_then_ppo", steps=total_steps)
    )
    result = dict(
        bc_validation=warm_validation,
        best_ppo_validation=best_validation,
        training_wall_s=time.perf_counter() - started,
        actual_steps=total_steps,
        demonstrations=len(labels),
    )
    (output / "training_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("Training finished:", json.dumps(result), flush=True)


def math_ceil(value):
    return int(np.ceil(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="omnidirectional/models/seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16384)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--teacher-episodes", type=int, default=32)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--validate-every", type=int, default=8)
    train(parser.parse_args())
