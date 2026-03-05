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
# Importações cruciais para normalização do ambiente
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Treinar agente PPO para PDCP slicing"
    )
    # [MANTER TODOS OS SEUS ARGUMENTOS AQUI. Excluí por brevidade, mas mantenha os seus]
    parser.add_argument("--config", type=str, default="src/environments/scenario_configurations/pdcp_use_case.json")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--ns3_path", type=str, default="/home/elioth/Documentos/artigo_jussi/ns-3-mmwave-oran")
    parser.add_argument("--total_timesteps", type=int, default=10000) # Reduzido para teste inicial
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    # n_steps reduzido para não esperar muito pelo ns-3 antes de atualizar a rede
    parser.add_argument("--n_steps", type=int, default=256) 
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="pdcp_ppo_final")
    parser.add_argument("--sla_embb_throughput", type=float, default=50.0)
    parser.add_argument("--sla_urllc_delay", type=float, default=5.0)
    parser.add_argument("--sla_miot_delivery_rate", type=float, default=0.99)
    parser.add_argument("--w_sla", type=float, default=0.7)
    parser.add_argument("--w_opt", type=float, default=0.3)

    args = parser.parse_args()

    # Carregar configuração
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', config_path)

    with open(config_path) as f:
        scenario_configuration = json.load(f)

    print("=" * 80)
    print("Treinamento do Agente DRL para PDCP Slicing")
    print("=" * 80)
    
    print("\nCriando ambiente PDCP Slicing...")
    # 1. Cria o ambiente Gym cru
    env_raw = PdcpSlicingEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=True,
        sla_embb_throughput=args.sla_embb_throughput,
        sla_urllc_delay=args.sla_urllc_delay,
        sla_miot_delivery_rate=args.sla_miot_delivery_rate,
        w_sla=args.w_sla,
        w_opt=args.w_opt,
    )

    # 2. Envolve em um DummyVecEnv (necessário para o VecNormalize)
    env = DummyVecEnv([lambda: env_raw])

    # 3. Aplica a Normalização nas Observações (e não na recompensa, para você conseguir debugar o valor real)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    print("Ambiente criado e Normalizado!")

    # Reset environment to get initial observation
    obs = env.reset()
    print(f"Primeira observação shape: {obs.shape}")

    # Configurar arquitetura da rede neural
    policy_kwargs = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=torch.nn.Tanh
    )

    print("\nInicializando modelo PPO...")
    model = PPO(
        "MlpPolicy",
        env, # Passa o ambiente já normalizado
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

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./checkpoints/",
        name_prefix="pdcp_ppo"
    )

    # Treinamento
    print(f"\nTreinando por {args.total_timesteps} timesteps...")
    print("=" * 80)

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback]
        )
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
    finally:
        # Salvar modelo final E os status de normalização
        model.save(args.model_name)
        env.save(os.path.join(args.output_folder, "vecnormalize.pkl"))
        print(f"\nModelo salvo em {args.model_name}.zip")
        print(f"Stats de normalização salvos em vecnormalize.pkl")

        # Fechar ambiente
        env.close()

    print("\nTreinamento concluído!")

if __name__ == '__main__':
    main()