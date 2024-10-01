from util import create_policy_1, create_policy_2, create_mdp
from model_free_prediction.monte_carlo_evaluator import MCEvaluator
from model_free_prediction.td_evaluator import TDEvaluator
from model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator

def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    mdp = create_mdp()

    policy_1 = create_policy_1()
    policy_2 = create_policy_2()

    mc_evaluator = MCEvaluator(env=mdp)

    # Evaluate policy 1 and 2 using Monte Carlo evaluation with 1000 episodes
    value_function_1 = mc_evaluator.evaluate(policy=policy_1, num_episodes=1000)
    print("Value Function for Policy 1 (MC):", value_function_1)

    value_function_2 = mc_evaluator.evaluate(policy=policy_2, num_episodes=1000)
    print("Value Function for Policy 2 (MC):", value_function_2)

    td_evaluator = TDEvaluator(env=mdp, alpha=0.1)

    # Evaluate the policies using TD(0) with 1000 episodes
    value_function_td_1 = td_evaluator.evaluate(policy=policy_1, num_episodes=1000)
    print("Value Function for Policy 1 (TD(0)):", value_function_td_1)

    value_function_td_2 = td_evaluator.evaluate(policy=policy_2, num_episodes=1000)
    print("Value Function for Policy 2 (TD(0)):", value_function_td_2)

    td_lambda_evaluator = TDLambdaEvaluator(env=mdp, alpha=0.1, lambd=0.5)

    # Evaluate the policy 1 and 2 using TD(λ) with 1000 episodes
    value_function_l_1 = td_lambda_evaluator.evaluate(policy=policy_1, num_episodes=1000)
    print("Value Function for Policy 1 (TD(λ)):", value_function_l_1)

    value_function_l_2 = td_lambda_evaluator.evaluate(policy=policy_2, num_episodes=1000)
    print("Value Function for Policy 2 (TD(λ)):", value_function_l_2)


if __name__ == "__main__":
    main()
