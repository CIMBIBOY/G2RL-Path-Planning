# ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class PPOAgent(nn.Module):
    def __init__(self, env, model, args, run_name, writer, wandb):
        super().__init__()
        self.env = env
        self.model = model
        self.args = args
        self.run_name = run_name
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, eps=1e-5)
        self.writer = writer
        self.wandb = wandb

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location = self.device, weights_only=True)
        print(checkpoint.keys())
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_model_only(self, path):
        checkpoint = torch.load(path, map_location = self.device, weights_only=True)
        print(checkpoint.keys())
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def update(self, obs, actions, masks, logprobs, advantages, returns, values, init_lstm, dones):
        # Optimizing the policy and value network
        assert self.args.num_envs % self.args.num_minibatches == 0
        envsperbatch = self.args.num_envs // self.args.num_minibatches
        envinds = np.arange(self.args.num_envs)
        flatinds = np.arange(self.args.batch_size).reshape(self.args.num_steps, self.args.num_envs)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, self.args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = self.model.get_action_and_value(
                    obs[mb_inds],
                    (init_lstm[0][:, mbenvinds], init_lstm[1][:, mbenvinds]),
                    dones[mb_inds],
                    masks[mb_inds],
                    actions.long()[mb_inds]
                )
                logratio = newlogprob - logprobs[mb_inds]
                # print(f"Logratio befor clamp: {logratio}")
                # logratio = torch.clamp(newlogprob - logprobs[mb_inds], -20, 20)
                # print(f"Logratio after clamp: {logratio}")
                ratio = logratio.exp()

                # Check for NaN or Inf values
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print("Warning: NaN or Inf detected in ratio")
                    print(f"ratio: {ratio}")
                    print(f"newlogprob: {newlogprob}")
                    print(f"logprobs: {logprobs[mb_inds]}")

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                '''
                # Logging intermediate values
                print(f"Epoch {epoch}, Batch {start//envsperbatch}")
                print(f"ratio mean: {ratio.mean().item():.4f}, min: {ratio.min().item():.4f}, max: {ratio.max().item():.4f}")
                print(f"logratio mean: {logratio.mean().item():.4f}, min: {logratio.min().item():.4f}, max: {logratio.max().item():.4f}")
                print(f"mb_advantages mean: {mb_advantages.mean().item():.4f}, std: {mb_advantages.std().item():.4f}")
                print(f"pg_loss1 mean: {pg_loss1.mean().item():.4f}, pg_loss2 mean: {pg_loss2.mean().item():.4f}")
                print(f"pg_loss: {pg_loss.item():.4f}")
                '''

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                '''
                # Logging value loss details
                print(f"v_loss: {v_loss.item():.4f}")
                print(f"newvalue mean: {newvalue.mean().item():.4f}, min: {newvalue.min().item():.4f}, max: {newvalue.max().item():.4f}")
                print(f"returns mean: {returns[mb_inds].mean().item():.4f}, min: {returns[mb_inds].min().item():.4f}, max: {returns[mb_inds].max().item():.4f}")
                #'''

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                '''
                # Logging total loss and its components
                print(f"entropy_loss: {entropy_loss.item():.4f}")
                print(f"total loss: {loss.item():.4f}")
                #'''

                self.optimizer.zero_grad()
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None:
                if approx_kl > self.args.target_kl:
                    print(f"Gradient norm ({total_norm:.2f}) exceeded threshold. Clipping to {self.args.max_grad_norm}")
                    break

        # Final logging for this update
        # print(f"Final approx_kl: {approx_kl:.4f}")
        # print(f"Final clipfracs: {np.mean(clipfracs):.4f}")
        
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return pg_loss.item(), v_loss.item(), entropy_loss.item(), old_approx_kl.item(), approx_kl.item(), clipfracs, explained_var

    def learn(self):
        num_updates = self.args.total_timesteps // self.args.batch_size
        
        # Initialize tensors
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.env.observation_space.shape[-4:]).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.env.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        action_masks = torch.zeros((self.args.num_steps, self.args.num_envs) + (5,)).to(self.device)
         
        global_step = 0
        start_time = time.time()

        # Reset the environment
        reset, info = self.env.reset()
        next_obs = torch.Tensor(reset).permute(1, 0, 2, 3, 4, 5).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        
        # Initialize LSTM states
        next_lstm_state = self.init_lstm_states(self.args.num_envs)

        '''
        print(f"actions shape: {actions.shape}")
        print(f"obs shape: {obs.shape}")
        print(f"logprobs shape: {logprobs.shape}")
        print(f"rewards shape: {rewards.shape}")
        print(f"dones shape: {dones.shape}")
        print(f"values shape: {values.shape}")  
        print(f"action_masks shape: {action_masks.shape}")  

        print(f"next_obs shape: {next_obs.shape}")
        print(f"next_done shape: {next_done.shape}")
        print(f"next_lstm_state shape: {[state.shape for state in next_lstm_state]}")
        #'''
        
        start_time = time.time()
        steps = 0
        batch_rewards = []

        for update in range(1, num_updates + 1):
            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.args.num_steps):
                global_step += 1 * self.args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                mask = torch.stack([env.action_mask(self.device) for env in self.env.envs])
                action_masks[step] = mask

                with torch.no_grad():
                    action, logprob, entropy, value, next_lstm_state = self.model.get_action_and_value(
                        next_obs, next_lstm_state, next_done, action_masks[step]
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # print(f"PPO action tensor: {action.cpu().numpy()}")
                next_obs, reward, done, trunc, info = self.env.step(action.cpu().numpy())
                
                '''
                # Debugging collisions
                mean_terminations_oc = np.zeros(4)
                for i in range(self.args.num_envs):
                    mean_terminations_oc[i] = self.env.envs[i].terminations[3]
                print(f"Collisions with obstacles in: - Env 1: {int(mean_terminations_oc[0])}, Env 2: {int(mean_terminations_oc[1])}, Env 3: {int(mean_terminations_oc[2])}, Env 4: {int(mean_terminations_oc[3])}\n")
                #'''

                termination_flags = np.logical_or(done, trunc)
                # print(f"Termination flags: {termination_flags}")
                steps += 1
                
                if self.args.pygame:
                    # Render the first environment instance 
                    self.env.envs[1].render()

                batch_rewards.append(reward)
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs = torch.Tensor(next_obs).permute(1, 0, 2, 3, 4, 5).to(self.device)
                # Convert the result to a tensor
                next_done = torch.Tensor(termination_flags).to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.model.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
                if self.args.gae:
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(self.device)
                    for t in reversed(range(self.args.num_steps)):
                        if t == self.args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + self.args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape[-4:]).unsqueeze(1)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_dones = dones.reshape(-1)
            b_action_masks = action_masks.reshape((-1, action_masks.shape[-1]))

            # Optimizing the policy and value network
            pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var = self.update(
                b_obs, b_actions, b_action_masks, b_logprobs, b_advantages, b_returns, b_values, initial_lstm_state, b_dones
            )

            mean_terminations_rg = np.zeros(1)
            mean_terminations_gi = np.zeros(1)
            mean_terminations_ms = np.zeros(1)
            mean_terminations_oc = np.zeros(4)

            curr_amr_count, curr_index = max((self.env.envs[i].amr_count, i) for i in range(4))

            for i in range(self.args.num_envs):
                mean_terminations_rg += self.env.envs[i].terminations[0]
                mean_terminations_gi += self.env.envs[i].terminations[1]
                mean_terminations_ms += self.env.envs[i].terminations[2]
                mean_terminations_oc[i] = self.env.envs[i].terminations[3]

            # Log to wandb
            if self.args.track:
                self.wandb.log({
                    "learning_rate": self.optimizer.param_groups[0]["lr"],  # Current learning rate used by the optimizer
                    "value_loss": v_loss,  # Loss related to how well the agent predicts the value of states
                    "policy_loss": pg_loss,  # Loss related to how well the agent learns to choose actions
                    "entropy_loss": entropy_loss,  # Entropy of the policy, indicating the randomness of action selection
                    "old_approx_kl": old_approx_kl,  # KL divergence between the old and new policy, indicating the change in policy
                    "approx_kl": approx_kl,  # Another measure of KL divergence to monitor policy updates
                    "clipfrac": np.mean(clipfracs),  # Fraction of updates where policy change was clipped to prevent large updates
                    "explained_variance": explained_var,  # Proportion of variance in returns explained by the value function
                    "SPS": int(global_step / (time.time() - start_time)),  # Steps Per Second, indicating the speed of training
                    "Reached goals": int(mean_terminations_rg),  # Count of episodes where the agent successfully reached the goal
                    "Lost guidance information": int(mean_terminations_gi),  # Count of episodes terminated due to loss of guidance
                    "Max steps reached": int(mean_terminations_ms),  # Count of episodes terminated due to reaching maximum steps
                    "Collisions with obstacles": int(mean_terminations_oc.sum()),  # Count of episodes terminated due to collisions
                    "Current max dynamic objects": curr_amr_count,  # Current number of dynamic objects, indicating environment complexity
                    "Global Steps": global_step,  # Total number of steps taken across all environments
                    "SPS": int(global_step / (time.time() - start_time)),  # Steps Per Second, reiterating the training speed
                })

            if self.args.target_kl is not None:
                if approx_kl > self.args.target_kl:
                    print(f"Early stopping upon to high Kullback-Leibler (KL) divergence value: {approx_kl}, target treshold: {self.args.target_kl}")
                    print(f" -------------------- Update: {update} -------------------- ")
                    print(f"Reward: {np.mean(batch_rewards):.4f}")
                    print(f"Policy Loss: {pg_loss:.4f}")
                    print(f"Value Loss: {v_loss:.4f}")
                    print(f"Entropy: {entropy_loss:.4f}")
                    print(f"KL Divergence: {approx_kl:.4f}")
                    print(f"Steps taken in {update} update: {steps}") ,  
                    print(f"Terminations casued by:\nReached goals: {int(mean_terminations_rg)}, No guidance information: {int(mean_terminations_gi)}, Max steps reached: {int(mean_terminations_ms)}\n")
                    print(f"Collisions with obstacles in:\nEnv 1: {int(mean_terminations_oc[0])}, Env 2: {int(mean_terminations_oc[1])}, Env 3: {int(mean_terminations_oc[2])}, Env 4: {int(mean_terminations_oc[3])}\n")
                    print(f"Current number of dynamic objects: {curr_amr_count} in env: {curr_index} (increasing based on curriculum learning)")
                    print(f"SPS (Steps Per Second): {int(global_step / (time.time() - start_time))}")
                    print(f" ---------------------------------------------------- ")
                    break

            # Update information
            if update % self.args.cmd_log == 0:
                end_time = time.time()
                computing_time = end_time - start_time
                print(f" -------------------- Update: {update} -------------------- ")
                print(f"Reward: {np.mean(batch_rewards):.4f}")
                print(f"Policy Loss: {pg_loss:.4f}")
                print(f"Value Loss: {v_loss:.4f}")
                print(f"Entropy: {entropy_loss:.4f}")
                print(f"KL Divergence: {approx_kl:.4f}")
                print(f"Computing time: {computing_time:.4f} s/{self.args.cmd_log} updates")
                print(f"Steps taken in {update} update: {steps}")
                print(f"Terminations casued by:\nReached goals: {int(mean_terminations_rg)}, No guidance information: {int(mean_terminations_gi)}, Max steps reached: {int(mean_terminations_ms)}\n")
                print(f"Collisions with obstacles in:\nEnv 1: {int(mean_terminations_oc[0])}, Env 2: {int(mean_terminations_oc[1])}, Env 3: {int(mean_terminations_oc[2])}, Env 4: {int(mean_terminations_oc[3])}\n")
                print(f"Current number of dynamic objects: {curr_amr_count} in env: {curr_index} (increasing based on curriculum learning)")
                print(f"SPS (Steps Per Second): {int(global_step / (time.time() - start_time))}")
                print(f" ---------------------------------------------------- ")
                start_time = time.time()
                steps = 0
                batch_rewards = []

                '''
                # TRY NOT TO MODIFY: record rewards for plotting purposes
                self.writer.add_scalar("logs/charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
                self.writer.add_scalar("logs/losses/value_loss", v_loss.item(), global_step)
                self.writer.add_scalar("logs/losses/policy_loss", pg_loss.item(), global_step)
                self.writer.add_scalar("logs/losses/entropy", entropy_loss.item(), global_step)
                self.writer.add_scalar("logs/losses/old_approx_kl", old_approx_kl.item(), global_step)
                self.writer.add_scalar("logs/losses/approx_kl", approx_kl.item(), global_step)
                self.writer.add_scalar("logs/losses/clipfrac", np.mean(clipfracs), global_step)
                self.writer.add_scalar("logs/losses/explained_variance", explained_var, global_step)
                self.writer.add_scalar("logs/charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                #'''

            if update % (self.args.cmd_log * 10) == 0:
                # Save model weights
                self.save(f'./eval/weights/{self.run_name}.pth')

        return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl
    
    def init_lstm_states(self, num_envs=1):
        return (
            torch.zeros(self.model.lstm.num_layers, num_envs, self.model.lstm.hidden_size).to(self.device),
            torch.zeros(self.model.lstm.num_layers, num_envs, self.model.lstm.hidden_size).to(self.device)
        )