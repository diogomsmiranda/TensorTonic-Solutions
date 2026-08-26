def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    next_values = []
    for state_i in range(len(values)):
        action_values = []
        for action_i in range(len(transitions[state_i])):
            expected_value = sum(probability * value for probability, value in zip(transitions[state_i][action_i], values))
            action_values.append(rewards[state_i][action_i] + gamma * expected_value)
        next_values.append(max(action_values))
    return next_values