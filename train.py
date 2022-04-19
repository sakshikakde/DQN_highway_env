from common_utils import *

def train_epoch(opt, em, agent, policy_net, target_net, memory, device, optimizer, criterion):
    em.reset()
    state = em.get_state()
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        
        if memory.can_provide_sample(opt.batch_size):
            experiences = memory.sample(opt.batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = ((next_q_values * opt.gamma) + rewards).type(torch.float32)

            loss = criterion(current_q_values, target_q_values.unsqueeze(1))
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if em.done:
            return(timestep)
