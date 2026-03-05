"""
Script de treinamento do agente DRL para controle de PDCP slicing.

Utiliza PPO (Proximal Policy Optimization) do stable-baselines3
para aprender a política ótima de split ratio por slice.

Baseado no artigo "RSLAQ - A Robust SLA-driven 6G O-RAN QoS xApp".
"""

import argparse
import json
import os
import sys

# Add src directory to path to import environments
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.pdcp_env import PdcpSlicingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Treinar agente PPO para PDCP slicing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/environments/scenario_configurations/pdcp_use_case.json",
        help="Path para arquivo de configuração"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Pasta de saída"
    )
    parser.add_argument(
        "--ns3_path",
        type=str,
        default="/home/elioth/Documentos/artigo_jussi/ns-3-mmwave-oran",
        help="Path para ns-3 mmWave O-RAN"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="Total de timesteps de treinamento"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Taxa de aprendizado"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Tamanho do batch"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Timesteps por update"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Número de épocas de otimização por update"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Fator de desconto"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Nível de verbosidade (0, 1, or 2)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pdcp_ppo_final",
        help="Nome do modelo salvo"
    )
    parser.add_argument(
        "--sla_embb_throughput",
        type=float,
        default=50.0,
        help="SLA de throughput para eMBB (Mbps)"
    )
    parser.add_argument(
        "--sla_urllc_delay",
        type=float,
        default=5.0,
        help="SLA de latência para URLLC (ms)"
    )
    parser.add_argument(
        "--sla_miot_delivery_rate",
        type=float,
        default=0.99,
        help="SLA de taxa de entrega para mIoT (0-1)"
    )
    parser.add_argument(
        "--w_sla",
        type=float,
        default=0.7,
        help="Peso para satisfação de SLA na função de recompensa"
    )
    parser.add_argument(
        "--w_opt",
        type=float,
        default=0.3,
        help="Peso para otimização de recursos na função de recompensa"
    )

    args = parser.parse_args()

    # Carregar configuração
    config_path = args.config
    if not os.path.isabs(config_path):
        # Make path relative to script location
        config_path = os.path.join(os.path.dirname(__file__), '..', config_path)

    with open(config_path) as f:
        scenario_configuration = json.load(f)

    print("=" * 80)
    print("Treinamento do Agente DRL para PDCP Slicing")
    print("=" * 80)
    print(f"Configuração: {config_path}")
    print(f"ns-3 path: {args.ns3_path}")
    print(f"Output folder: {args.output_folder}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"N steps: {args.n_steps}")
    print(f"N epochs: {args.n_epochs}")
    print(f"Gamma: {args.gamma}")
    print(f"SLA eMBB throughput: {args.sla_embb_throughput} Mbps")
    print(f"SLA URLLC delay: {args.sla_urllc_delay} ms")
    print(f"SLA mIoT delivery rate: {args.sla_miot_delivery_rate}")
    print(f"Reward weights: w_sla={args.w_sla}, w_opt={args.w_opt}")
    print("=" * 80)

    print("\nCriando ambiente PDCP Slicing...")
    env = PdcpSlicingEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=True,
        verbose=(args.verbose >= 2),
        sla_embb_throughput=args.sla_embb_throughput,
        sla_urllc_delay=args.sla_urllc_delay,
        sla_miot_delivery_rate=args.sla_miot_delivery_rate,
        w_sla=args.w_sla,
        w_opt=args.w_opt,
    )

    print("Ambiente criado!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Reset environment to get initial observation
    obs, info = env.reset()
    print(f"Primeira observação shape: {obs.shape}")

    # Configurar arquitetura da rede neural
    policy_kwargs = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Arquitetura MLP
        activation_fn=torch.nn.Tanh
    )

    print("\nInicializando modelo PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        tensorboard_log="./tensorboard/"
    )

    # Callbacks para salvar checkpoints e avaliação
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./checkpoints/",
        name_prefix="pdcp_ppo"
    )

    eval_callback = EvalCallback(
        env,
        eval_freq=10000,
        n_eval_episodes=3,
        deterministic=True,
        verbose=args.verbose,
        best_model_save_path="./best_model/"
    )

    # Treinamento
    print(f"\nTreinando por {args.total_timesteps} timesteps...")
    print("=" * 80)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback]
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
    finally:
        # Salvar modelo final
        model.save(args.model_name)
        print(f"\nModelo salvo em {args.model_name}.zip")

        # Fechar ambiente
        env.close()

    print("\nTreinamento concluído!")
    print("=" * 80)


if __name__ == '__main__':
    main()
