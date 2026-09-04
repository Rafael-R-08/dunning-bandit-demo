"""
run_experiment.py

Compara duas políticas de escolha de playbook de dunning ao longo de N episódios:

    1. Baseline fixa: usa sempre o mesmo playbook para todos os contextos
       (imita a prática comum descrita no projeto: "retry em dias 1-3-7,
       mensagem genérica para todos").

    2. Thompson Sampling contextual: mantém uma distribuição Beta(alpha, beta)
       por (contexto, playbook) e, em cada episódio, amostra dessa distribuição
       para decidir qual playbook experimentar. É "contextual" porque as
       distribuições são mantidas separadamente por contexto (bucketing),
       em vez de uma única distribuição global por playbook.

Métrica principal: taxa de recuperação acumulada ao longo do tempo.

Isto é uma demonstração pedagógica do mecanismo de exploration/exploitation
em bandits contextuais, não uma implementação de produção nem o simulador
completo pedido no projeto original.
"""

import numpy as np
import matplotlib.pyplot as plt

from dunning_env import DunningEnv, PLAYBOOKS, FAILURE_CODES, SEGMENTS

N_EPISODES = 5000
SEED = 42


def run_fixed_baseline(env: DunningEnv, playbook_idx: int, n_episodes: int):
    """Política ingénua: usa sempre o mesmo playbook, independentemente do contexto."""
    rewards = np.zeros(n_episodes, dtype=int)
    for t in range(n_episodes):
        context = env.sample_context()
        rewards[t] = env.step(context, playbook_idx)
    return rewards


def run_thompson_sampling(env: DunningEnv, n_episodes: int):
    """Thompson Sampling contextual com prior Beta(1,1) por (contexto, playbook)."""
    n_arms = len(PLAYBOOKS)

    # alpha/beta por contexto: dicionário context_key -> array [n_arms]
    alpha = {}
    beta = {}

    def ensure_context(key):
        if key not in alpha:
            alpha[key] = np.ones(n_arms)
            beta[key] = np.ones(n_arms)

    rewards = np.zeros(n_episodes, dtype=int)
    chosen_arms = np.zeros(n_episodes, dtype=int)

    for t in range(n_episodes):
        context = env.sample_context()
        key = env.context_key(context)
        ensure_context(key)

        # Amostra uma probabilidade de sucesso "acreditada" por braço e escolhe a maior
        sampled = np.random.beta(alpha[key], beta[key])
        arm = int(np.argmax(sampled))

        reward = env.step(context, arm)

        # Atualiza a crença (posterior Beta) do braço escolhido, neste contexto
        alpha[key][arm] += reward
        beta[key][arm] += 1 - reward

        rewards[t] = reward
        chosen_arms[t] = arm

    return rewards, chosen_arms, alpha, beta


def cumulative_rate(rewards: np.ndarray) -> np.ndarray:
    return np.cumsum(rewards) / (np.arange(len(rewards)) + 1)


def main():
    np.random.seed(SEED)
    env = DunningEnv(seed=SEED)

    # Baseline: playbook fixo "retry_delayed_msg" (índice 1), um dos mais
    # razoáveis "em média" mas subótimo em vários contextos específicos.
    baseline_idx = 1
    baseline_rewards = run_fixed_baseline(
        DunningEnv(seed=SEED), baseline_idx, N_EPISODES
    )

    ts_rewards, chosen_arms, alpha, beta = run_thompson_sampling(
        DunningEnv(seed=SEED), N_EPISODES
    )

    baseline_cum = cumulative_rate(baseline_rewards)
    ts_cum = cumulative_rate(ts_rewards)

    print("=== Resultado final (taxa de recuperação acumulada) ===")
    print(f"Baseline fixa ('{PLAYBOOKS[baseline_idx]}'): {baseline_cum[-1]:.3f}")
    print(f"Thompson Sampling contextual:               {ts_cum[-1]:.3f}")
    print()
    print("=== Playbook mais escolhido por contexto (após aprendizagem) ===")
    for fc in FAILURE_CODES:
        for seg in SEGMENTS:
            key = f"{fc}|{seg}"
            if key in alpha:
                best_arm = int(np.argmax(alpha[key] / (alpha[key] + beta[key])))
                print(f"  {fc:20s} / {seg:3s} -> {PLAYBOOKS[best_arm]}")

    # Gráfico comparativo
    plt.figure(figsize=(9, 5))
    plt.plot(baseline_cum, label=f"Baseline fixa ({PLAYBOOKS[baseline_idx]})", linewidth=2)
    plt.plot(ts_cum, label="Thompson Sampling contextual", linewidth=2)
    plt.xlabel("Episódio (pagamento falhado)")
    plt.ylabel("Taxa de recuperação acumulada")
    plt.title("Baseline fixa vs. Thompson Sampling contextual\n(simulador simplificado de dunning)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nGráfico guardado em results.png")


if __name__ == "__main__":
    main()
