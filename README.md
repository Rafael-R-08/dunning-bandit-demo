# Contextual Thompson Sampling para Recuperação de Pagamentos (Dunning) — Demo

Pequeno projeto exploratório, feito no âmbito da preparação de candidatura ao
projeto de investigação **"Adaptive Methods for Payment Recovery (Dunning)"**
(IPV), cuja área de investigação é a tomada de decisão estatística para
recuperação de receita de subscrições usando *contextual bandits* e
*reinforcement learning* de curto horizonte.

## Objetivo deste repositório

Este **não** é o simulador de dunning pedido no projeto (que exigiria dados
reais, uma modelação muito mais rica de retries, incentivos, budgets,
compliance, proxies de complaint/chargeback, etc.). É antes um exercício
pessoal, em escala reduzida, para:

1. Consolidar a intuição sobre *contextual bandits* aplicada especificamente
   ao problema de dunning descrito no projeto;
2. Comparar, de forma simples, uma política estática tipo "regra fixa"
   (comum na prática, segundo o próprio documento do projeto) com uma
   política adaptativa que aprende por contexto;
3. Servir como ponto de partida de conversa técnica na candidatura/entrevista.

## O que está implementado

- **`dunning_env.py`** — Simulador sintético minimalista: cada episódio é um
  pagamento falhado com um contexto (`failure_code`, `segment`) e existe um
  pequeno conjunto de *playbooks* (estratégias de recuperação) com
  probabilidades de sucesso diferentes consoante o contexto. As
  probabilidades "verdadeiras" são inventadas — servem só para dar estrutura
  ao problema.

- **`run_experiment.py`** — Duas políticas:
  - **Baseline fixa**: usa sempre o mesmo playbook para todos os contextos
    (imita a prática descrita no projeto: "retry em dias 1-3-7, mensagem
    genérica para todos").
  - **Thompson Sampling contextual**: mantém uma distribuição
    `Beta(alpha, beta)` por par (contexto, playbook), amostrando dela em
    cada episódio para decidir qual playbook experimentar, e atualizando a
    crença com o resultado observado (recuperado / não recuperado).

- **`results.png`** — Gráfico da taxa de recuperação acumulada ao longo de
  5000 episódios simulados, comparando as duas políticas.

## Resultado

Numa corrida de 5000 episódios simulados:

| Política                         | Taxa de recuperação acumulada |
|-----------------------------------|:------------------------------:|
| Baseline fixa (`retry_delayed_msg`) | ~0.39                         |
| Thompson Sampling contextual        | ~0.49                         |

Mais relevante do que o número em si: a política de Thompson Sampling
**aprende, sem lhe dizer nada à partida**, que `expired_card` responde melhor
a `update_payment_msg` e que o segmento `b2b` responde melhor a
`personal_outreach` — que é exatamente a estrutura (arbitrária) que defini
no simulador. Isto ilustra o mecanismo central do projeto: substituir regras
rígidas por políticas que aprendem qual estratégia funciona para qual
situação.

## Limitações (deliberadas, dado o âmbito deste exercício)

- Não há budget/restrições por episódio (número máximo de retries,
  contactos ou incentivos) — o projeto real trata isto como uma restrição
  explícita (ver referência a *Bandits with Knapsacks*, Badanidiyuru et al.,
  nas referências do projeto).
- Não há proxy de complaint/chargeback nem custo de incentivo — o projeto
  real pretende que a política não piore estes indicadores.
- O contexto é tratado por *bucketing* discreto (uma distribuição Beta por
  combinação de contexto), não por um modelo linear/contínuo de contexto
  como seria necessário com features como tenure, engagement ou histórico
  de dunning.
- Não há avaliação offline (doubly robust ou similar) — o projeto real
  refere isto explicitamente (Dudík, Langford & Li, 2011) como parte da
  metodologia de avaliação de políticas.
- As probabilidades de sucesso são inventadas, não vêm de dados reais.

## Como correr

```bash
pip install numpy matplotlib
python run_experiment.py
```

## Referências que motivaram as escolhas de design

- Agrawal, S., e Goyal, N. (2013). *Thompson Sampling for Contextual Bandits
  with Linear Payoffs*. ICML.
- Badanidiyuru, A., Kleinberg, R., e Slivkins, A. (2013). *Bandits with
  Knapsacks*. FOCS.
- Dudík, M., Langford, J., e Li, L. (2011). *Doubly Robust Policy Evaluation
  and Learning*. ICML.
- Lattimore, T., e Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge
  University Press.
