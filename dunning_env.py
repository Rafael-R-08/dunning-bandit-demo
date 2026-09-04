"""
dunning_env.py

Simulador sintético e simplificado de episódios de "dunning" (recuperação de
pagamentos falhados), inspirado no documento de projeto "Adaptive Methods for
Payment Recovery (Dunning)".

Este NÃO é o simulador completo pedido no projeto (que envolve retries,
comunicação, incentivos, budgets, proxies de complaint/chargeback, etc.).
É uma versão minimalista, feita para demonstrar o raciocínio central:

    - Cada pagamento falhado é um "episódio" com um CONTEXTO observável
      (motivo da falha, segmento do cliente).
    - Existe um pequeno conjunto de PLAYBOOKS (estratégias de recuperação)
      para escolher, cada um representado como um "braço" (arm) do bandit.
    - A probabilidade de recuperação depende da combinação (contexto, playbook).
    - O objetivo é aprender, ao longo de muitos episódios, qual playbook
      funciona melhor para cada contexto, sem o saber à partida.

Motivos de falha (failure_code):
    - "insufficient_funds": recuperação melhora com retry adiado (paydays)
    - "expired_card": recuperação melhora com pedido explícito de atualização
    - "generic_decline": comportamento mais neutro/ambíguo

Segmentos (segment):
    - "b2c": mais sensível a mensagens simples e incentivos pequenos
    - "b2b": mais sensível a contacto direto / prazos alargados

Playbooks (as ações / braços do bandit):
    0 - "retry_immediate"      -> retry rápido, sem mensagem extra
    1 - "retry_delayed_msg"    -> retry adiado + mensagem simples
    2 - "update_payment_msg"   -> pedido explícito de atualização de cartão
    3 - "personal_outreach"    -> contacto mais direto (ex.: e-mail dedicado B2B)

As probabilidades de sucesso abaixo são inventadas para fins pedagógicos,
não vêm de dados reais.
"""

import numpy as np

FAILURE_CODES = ["insufficient_funds", "expired_card", "generic_decline"]
SEGMENTS = ["b2c", "b2b"]
PLAYBOOKS = [
    "retry_immediate",
    "retry_delayed_msg",
    "update_payment_msg",
    "personal_outreach",
]

# Probabilidade "verdadeira" de recuperação por (failure_code, segment, playbook).
# Estes números não são reais -- servem só para o simulador ter estrutura,
# ou seja, para existir de facto um playbook melhor por contexto que a
# política tem de aprender a descobrir.
_TRUE_PROBS = {
    ("insufficient_funds", "b2c"): [0.25, 0.55, 0.20, 0.30],
    ("insufficient_funds", "b2b"): [0.30, 0.50, 0.25, 0.45],
    ("expired_card", "b2c"):       [0.15, 0.30, 0.65, 0.35],
    ("expired_card", "b2b"):       [0.20, 0.30, 0.55, 0.50],
    ("generic_decline", "b2c"):    [0.30, 0.35, 0.35, 0.30],
    ("generic_decline", "b2b"):    [0.30, 0.30, 0.30, 0.45],
}


class DunningEnv:
    """Ambiente simplificado: gera episódios e devolve recompensa binária
    (1 = pagamento recuperado, 0 = não recuperado) para o playbook escolhido.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def sample_context(self):
        """Gera o contexto observável de um novo episódio de falha de pagamento."""
        failure_code = self.rng.choice(FAILURE_CODES)
        segment = self.rng.choice(SEGMENTS)
        return {"failure_code": failure_code, "segment": segment}

    def step(self, context: dict, playbook_idx: int) -> int:
        """Dado um contexto e o playbook escolhido, devolve 1 (recuperado)
        ou 0 (não recuperado), amostrado da probabilidade verdadeira."""
        key = (context["failure_code"], context["segment"])
        p = _TRUE_PROBS[key][playbook_idx]
        return int(self.rng.random() < p)

    @staticmethod
    def context_key(context: dict) -> str:
        """Chave usada pelas políticas para indexar o contexto (bucketing)."""
        return f"{context['failure_code']}|{context['segment']}"
