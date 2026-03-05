from typing_extensions import override
import numpy as np
import pandas as pd
import gymnasium as gym
import glob
import csv
import os
from typing import Any, SupportsFloat, Tuple, Dict, List

from nsoran.ns_env import NsOranEnv

# Reward function components (RSLAQ PDCP Slicing)
# 'THROUGHPUT_embb' represents the mean pdcpThroughput for eMBB UEs.
# 'DELAY_urllc' represents the mean pdcpLatency for URLLC UEs.
# 'DELIVERY_RATE_miot' represents the estimated delivery rate based on Buffer size for mIoT UEs.
#
# Reward function formula:
# r = Σ_s (w_sla * SLA_satisfaction_s - w_opt * split_penalty_s)
# Where:
# - SLA_satisfaction: 1.0 if SLA met, 0.0 otherwise
# - split_penalty: Current split ratio (higher LTE split = penalty)

class PdcpSlicingEnv(NsOranEnv):
    
    grafana_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "step": "INTEGER",
        "thp_embb": "REAL",
        "thp_urllc": "REAL",
        "thp_miot": "REAL",
        "del_embb": "REAL",
        "del_urllc": "REAL",
        "del_miot": "REAL",
        "buf_embb": "REAL",
        "buf_urllc": "REAL",
        "buf_miot": "REAL",
        "split_embb": "REAL",
        "split_urllc": "REAL",
        "split_miot": "REAL",
        "reward": "REAL"
    }
        
    def __init__(
        self, 
        ns3_path: str, 
        scenario_configuration: dict, 
        output_folder: str, 
        optimized: bool,
        sla_embb_throughput: float = 50.0,
        sla_urllc_delay: float = 5.0,
        sla_miot_delivery_rate: float = 0.99,
        w_sla: float = 0.7,
        w_opt: float = 0.3,
    ):
        super().__init__(
            ns3_path=ns3_path, 
            scenario='scenario-two', 
            scenario_configuration=scenario_configuration,
            output_folder=output_folder, 
            optimized=optimized,
            control_header=['timestamp', 'ueId', 'percentage'], 
            log_file='QosActions.txt', 
            control_file='qos_actions.csv'
        )
        
        self.folder_name = "Simulation"
        self.ns3_simulation_time = scenario_configuration['simTime'][0] * 1000
        
        # Slices Configuration
        self.slices = ['embb', 'urllc', 'miot']
        self.sla_embb_throughput = sla_embb_throughput
        self.sla_urllc_delay = sla_urllc_delay
        self.sla_miot_delivery_rate = sla_miot_delivery_rate
        self.w_sla = w_sla
        self.w_opt = w_opt

        perc_embb = scenario_configuration.get('PercUEeMBB', [0.33])
        perc_urllc = scenario_configuration.get('PercUEURLLC', [0.33])
        self.perc_embb = perc_embb[0] if isinstance(perc_embb, list) else perc_embb
        self.perc_urllc = perc_urllc[0] if isinstance(perc_urllc, list) else perc_urllc
        self.perc_miot = 1.0 - self.perc_embb - self.perc_urllc

        ues_config = scenario_configuration.get('ues', [30])
        self.total_ues = ues_config[0] if isinstance(ues_config, list) else ues_config
        
        # Action space: Continuous split ratio for 3 slices (0.0 to 1.0)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 12 values (thp, delay, buffer, split ratio for 3 slices)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.columns_state = [
            'THP_embb', 'DEL_embb', 'BUF_embb', 'SPLIT_embb',
            'THP_urllc', 'DEL_urllc', 'BUF_urllc', 'SPLIT_urllc',
            'THP_miot', 'DEL_miot', 'BUF_miot', 'SPLIT_miot'
        ]

        self.last_actions = {s: 0.5 for s in self.slices}
        self.observations = pd.DataFrame()
        self.num_steps = 0
        
        self._compute_slice_ue_indices()

    def _compute_slice_ue_indices(self):
        """Compute which UEs belong to each slice based on scenario-two.cc."""
        total_ues = self.total_ues
        embb_end = int(total_ues * self.perc_embb)
        urllc_end = int(total_ues * (self.perc_embb + self.perc_urllc))

        self.slice_ue_indices = {
            'embb': list(range(0, embb_end)),
            'urllc': list(range(embb_end, urllc_end)),
            'miot': list(range(urllc_end, total_ues))
        }
        
        self.ue_slice_map = {}
        for idx in self.slice_ue_indices['embb']: self.ue_slice_map[idx + 1] = 'embb'
        for idx in self.slice_ue_indices['urllc']: self.ue_slice_map[idx + 1] = 'urllc'
        for idx in self.slice_ue_indices['miot']: self.ue_slice_map[idx + 1] = 'miot'

    @override
    def _compute_action(self, action):
        """Converts the agents action into the ns-O-RAN required format"""
        action = np.clip(action, 0.0, 1.0)
        cell_act_comb_lst = []

        for idx, (slice_name, split_ratio) in enumerate(zip(self.slices, action)):
            ue_indices = self.slice_ue_indices.get(slice_name, [])
            for ue_idx in ue_indices:
                ue_imsi = ue_idx + 1
                cell_act_comb_lst.append([int(ue_imsi), float(split_ratio)])
            
            self.last_actions[slice_name] = split_ratio

        return cell_act_comb_lst

    @override
    def _get_obs(self):
        """Queries the datalake, processes into Pandas DF and returns state array"""
        kpms_raw = [
            "DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)", 
            "DRB.PdcpSduDelayDl.UEID (pdcpLatency)",      
            "DRB.BufferSize.Qos.UEID"                     
        ]       
        
        # Read from Datalake
        ue_kpms = self.datalake.read_kpms(self.last_timestamp, kpms_raw) 
        
        if not ue_kpms:
            # Fallback if no data is present yet
            df = pd.DataFrame(columns=['ueImsiComplete', 'pdcpThroughput', 'pdcpLatency', 'BufferSize'])
        else:
            columns = ['ueImsiComplete', 'pdcpThroughput', 'pdcpLatency', 'BufferSize']
            df = pd.DataFrame(ue_kpms, columns=columns)
            
        df["timestamp"] = self.last_timestamp
        
        # Clean up types and map slices
        df['pdcpThroughput'] = pd.to_numeric(df['pdcpThroughput'], errors='coerce').fillna(0.0)
        df['pdcpLatency'] = pd.to_numeric(df['pdcpLatency'], errors='coerce').fillna(0.0)
        df['BufferSize'] = pd.to_numeric(df['BufferSize'], errors='coerce').fillna(0.0)
        df['slice'] = df['ueImsiComplete'].map(self.ue_slice_map)
        
        # Transform from UE-centric to Slice-centric
        self.observations = self.ue_centric_toslice_centric(df)
        
        # Extract the state tuple
        states = self.observations[self.columns_state]
        states_array = states.iloc[0].values.astype(np.float32)
        
        return states_array
        
    @override
    def _compute_reward(self):
        """Calculates RSLAQ reward and logs to Grafana table in Datalake"""
        cell_df = self.observations.copy()
        reward = 0.0
        
        # Define metrics for easy access
        metrics = {
            'embb': {'thp': cell_df['THP_embb'].iloc[0], 'del': cell_df['DEL_embb'].iloc[0], 'buf': cell_df['BUF_embb'].iloc[0], 'split': cell_df['SPLIT_embb'].iloc[0]},
            'urllc': {'thp': cell_df['THP_urllc'].iloc[0], 'del': cell_df['DEL_urllc'].iloc[0], 'buf': cell_df['BUF_urllc'].iloc[0], 'split': cell_df['SPLIT_urllc'].iloc[0]},
            'miot': {'thp': cell_df['THP_miot'].iloc[0], 'del': cell_df['DEL_miot'].iloc[0], 'buf': cell_df['BUF_miot'].iloc[0], 'split': cell_df['SPLIT_miot'].iloc[0]}
        }

        # Calculate Delivery Rate for mIoT based on buffer size (simplified heuristic)
        delivery_rate_miot = max(0.0, 1.0 - (metrics['miot']['buf'] / 10000)) if metrics['miot']['buf'] > 1000 else 1.0
        
        # Check SLAs
        sla_embb_ok = 1.0 if metrics['embb']['thp'] >= self.sla_embb_throughput else 0.0
        sla_urllc_ok = 1.0 if metrics['urllc']['del'] <= self.sla_urllc_delay else 0.0
        sla_miot_ok = 1.0 if delivery_rate_miot >= self.sla_miot_delivery_rate else 0.0

        # RSLAQ Reward Formula per slice
        r_embb = self.w_sla * sla_embb_ok - self.w_opt * metrics['embb']['split']
        r_urllc = self.w_sla * sla_urllc_ok - self.w_opt * metrics['urllc']['split']
        r_miot = self.w_sla * sla_miot_ok - self.w_opt * metrics['miot']['split']
        
        reward = r_embb + r_urllc + r_miot
        
        # Extra penalty if multiple SLAs fail
        if (sla_embb_ok + sla_urllc_ok + sla_miot_ok) < 2:
            reward -= 0.5
            
        self.num_steps += 1

        # Save to Grafana Database
        db_row = {
            'timestamp': int(self.last_timestamp),
            'ueImsiComplete': None,
            'step': self.num_steps,
            'thp_embb': float(metrics['embb']['thp']),
            'thp_urllc': float(metrics['urllc']['thp']),
            'thp_miot': float(metrics['miot']['thp']),
            'del_embb': float(metrics['embb']['del']),
            'del_urllc': float(metrics['urllc']['del']),
            'del_miot': float(metrics['miot']['del']),
            'buf_embb': float(metrics['embb']['buf']),
            'buf_urllc': float(metrics['urllc']['buf']),
            'buf_miot': float(metrics['miot']['buf']),
            'split_embb': float(metrics['embb']['split']),
            'split_urllc': float(metrics['urllc']['split']),
            'split_miot': float(metrics['miot']['split']),
            'reward': float(reward)
        }
        
        if hasattr(self, 'datalake') and self.datalake is not None:
            self.datalake.insert_data("grafana", db_row)
            
        return reward

    @override
    def _init_datalake_usecase(self):
        self.datalake._create_table("grafana", self.grafana_keys)  
        return super()._init_datalake_usecase()

    @override
    def _fill_datalake_usecase(self):
        """Reads NS-3 trace files and pushes them to SQLite datalake"""
        if not hasattr(self, 'datalake') or self.datalake is None:
            return

        timestamp = int(self.last_timestamp)
        sim_dir = getattr(self, 'simulation_dir', self.output_folder)

        file_patterns = {
            'cu_up_*': self.datalake.insert_lte_cu_up,  
            'du_*': self.datalake.insert_du             
        }

        for pattern, insert_func in file_patterns.items():
            search_path = os.path.join(sim_dir, f"{pattern}")
            
            for filepath in glob.glob(f"{search_path}.csv") + glob.glob(f"{search_path}.txt"):
                try:
                    with open(filepath, 'r') as f:
                        reader = csv.DictReader(f, delimiter=',')
                        for row in reader:
                            if 'timestamp' not in row: continue
                            try:
                                row_ts = int(float(row['timestamp']))
                            except ValueError:
                                continue
                            
                            if row_ts == timestamp:
                                clean_row = {}
                                for key, value in row.items():
                                    clean_key = key.strip()
                                    try:
                                        f_val = float(value)
                                        clean_row[clean_key] = int(f_val) if f_val.is_integer() else f_val
                                    except (ValueError, TypeError):
                                        clean_row[clean_key] = value
                                insert_func(clean_row)
                except Exception as e:
                    pass

    def ue_centric_toslice_centric(self, df):
        """Groups UE KPMs by slice to calculate aggregated metrics and pivots to a flat state row"""
        
        # Aggregate by slice
        if not df.empty and 'slice' in df.columns:
            slice_df = df.groupby('slice').agg({
                'pdcpThroughput': 'mean',
                'pdcpLatency': 'mean',
                'BufferSize': 'mean'
            }).reset_index()
        else:
            slice_df = pd.DataFrame(columns=['slice', 'pdcpThroughput', 'pdcpLatency', 'BufferSize'])
            
        # Create a single row DataFrame for the model observations
        flat_state = {}
        for s in self.slices:
            # Get values or default to 0.0 if slice missing
            slice_data = slice_df[slice_df['slice'] == s]
            thp = slice_data['pdcpThroughput'].values[0] if not slice_data.empty else 0.0
            del_ = slice_data['pdcpLatency'].values[0] if not slice_data.empty else 0.0
            buf = slice_data['BufferSize'].values[0] if not slice_data.empty else 0.0
            
            flat_state[f'THP_{s}'] = float(thp)
            flat_state[f'DEL_{s}'] = float(del_)
            flat_state[f'BUF_{s}'] = float(buf)
            flat_state[f'SPLIT_{s}'] = float(self.last_actions[s])
            
        return pd.DataFrame([flat_state])