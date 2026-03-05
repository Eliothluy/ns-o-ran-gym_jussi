import argparse
import json
from environments.pdcp_env import PdcpSlicingEnv

if __name__ == '__main__':
    #######################
    # Parse arguments #
    #######################
    parser = argparse.ArgumentParser(description="Run the PDCP Slicing environment (RSLAQ)")
    parser.add_argument("--config", type=str, default="/home/elioth/Documentos/artigo_jussi/ns-o-ran-gym_jussi/src/environments/scenario_configurations/pdcp_use_case.json",
                        help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/home/elioth/Documentos/artigo_jussi/ns-3-mmwave-oran/",
                        help="Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of steps to run in the environment")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")

    args = parser.parse_args()

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    num_steps = args.num_steps
    optimized = args.optimized

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        exit(-1)

    scenario_configuration = json.loads(params)

    print('Creating PDCP Slicing Environment')
    env = PdcpSlicingEnv(ns3_path=ns3_path, scenario_configuration=scenario_configuration, 
                         output_folder=output_folder, optimized=optimized)

    print('Environment Created!')

    print('Launch reset ', end='', flush=True)
    obs, info = env.reset()
    print('done')
    
    print(f'First set of observations {obs}')
    print(f'Info {info}')

    # Action logic
    # Neste loop, simulamos uma ação fixa ou aleatória para testar a comunicação
    for step in range(2, num_steps):
        model_action = []
        
        # --- Lógica de Ação ---
        # No exemplo ES, ele lia estados e agia. 
        # Para PDCP Slicing, a ação é o Split Ratio para cada slice (eMBB, URLLC, mIoT).
        # Como teste, vamos manter um split fixo (ex: 50% para cada) ou aleatório.
        
        # Gera 3 valores de ação (um para cada slice) entre 0.0 e 1.0
        # model_action deve ter shape (3,) para o nosso ambiente
        
        # Exemplo: Ação estática para teste (50% LTE / 50% mmWave)
        model_action = [0.5, 0.5, 0.5] 
        
        # Ou ação aleatória (descomente para variar):
        # import numpy as np
        # model_action = np.random.uniform(low=0.0, high=1.0, size=(3,)).tolist()
        
        print(f'Step {step} ', end='', flush=True)
        
        # O método step espera a ação. 
        # O retorno 'truncated' indica se o episódio terminou por limite de tempo
        obs, reward, terminated, truncated, info = env.step(model_action)

        print('done', flush=True)

        print(f'Status t = {step}')
        # Exibe a ação computada (convertida para formato do ns-3)
        print(f'Actions {env._compute_action(model_action)}') 
        print(f'Observations {obs}')
        print(f'Reward {reward}')
        print(f'Terminated {terminated}')
        print(f'Truncated {truncated}')
        print(f'Info {info}')

        # If the environment is over, exit
        if terminated:
            break

        # If the episode is up (environment still running), then start another one
        if truncated:
            break # We don't want this outside the training
            obs, info = env.reset() 
