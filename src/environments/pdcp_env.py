"""
PdcpSlicingEnv - Environment for PDCP Slicing control in Dual Connectivity (EN-DC).

Based on the article "RSLAQ - A Robust SLA-driven 6G O-RAN QoS xApp".

This environment manages the split ratio between LTE (eNB) and NR (gNB) for three
network slices (eMBB, URLLC, mIoT) at the PDCP layer.
"""

import csv
import os
import gymnasium as gym
import numpy as np
from typing import Any, SupportsFloat, Tuple, Dict, List

from nsoran.ns_env import NsOranEnv


class PdcpSlicingEnv(NsOranEnv):
    """
    Environment Gym para controle de PDCP Slicing em Dual Connectivity (EN-DC).

    Espaço de Ação: Box(3, dtype=np.float32)
        - Ação[0]: qos_embb - Split ratio para eMBB (0.0=todo NR, 1.0=todo LTE)
        - Ação[1]: qos_urllc - Split ratio para URLLC
        - Ação[2]: qos_miot - Split ratio para mIoT

    Espaço de Observação: Box(12, dtype=np.float32)
        - Métricas por slice: throughput médio, latência média, buffer status, split ratio atual
        - N = 12 (4 métricas x 3 slices)

    Recompensa: Baseada em SLA violations e otimização de recursos (artigo RSLAQ)
    """

    def __init__(
        self,
        ns3_path: str,
        scenario_configuration: dict,
        output_folder: str,
        optimized: bool,
        verbose: bool = False,
        sla_embb_throughput: float = 50.0,
        sla_urllc_delay: float = 5.0,
        sla_miot_delivery_rate: float = 0.99,
        w_sla: float = 0.7,
        w_opt: float = 0.3,
    ):
        """
        Initialize the PDCP Slicing environment.

        Args:
            ns3_path: Path to the ns-3 folder.
            scenario_configuration: Dictionary containing simulation parameters.
            output_folder: Output folder for simulation data.
            optimized: If True, run ns-3 in optimized mode.
            verbose: Enable verbose logging.
            sla_embb_throughput: Minimum throughput for eMBB slice (Mbps).
            sla_urllc_delay: Maximum delay for URLLC slice (ms).
            sla_miot_delivery_rate: Minimum delivery rate for mIoT slice (0-1).
            w_sla: Weight for SLA satisfaction in reward function.
            w_opt: Weight for resource optimization in reward function.
        """
        super().__init__(
            ns3_path=ns3_path,
            scenario='scenario-two',
            scenario_configuration=scenario_configuration,
            output_folder=output_folder,
            optimized=optimized,
            control_header=['timestamp', 'ueId', 'percentage'],
            log_file='QosActions.txt',
            control_file='qos_actions_for_ns3.csv'
        )

        # Configuration specific to scenario-two
        self.folder_name = "Simulation"
        self.ns3_simulation_time = scenario_configuration['simTime'] * 1000

        # Slice definitions
        self.slices = ['embb', 'urllc', 'miot']
        self.verbose = verbose

        # SLA thresholds
        self.sla_embb_throughput = sla_embb_throughput
        self.sla_urllc_delay = sla_urllc_delay
        self.sla_miot_delivery_rate = sla_miot_delivery_rate

        # Reward function weights (RSLAQ paper)
        self.w_sla = w_sla  # Weight for SLA satisfaction
        self.w_opt = w_opt  # Weight for resource optimization

        # Get slice percentages from configuration
        perc_embb = scenario_configuration.get('PercUEeMBB', [0.33])
        perc_urllc = scenario_configuration.get('PercUEURLLC', [0.33])
        self.perc_embb = perc_embb[0] if isinstance(perc_embb, list) else perc_embb
        self.perc_urllc = perc_urllc[0] if isinstance(perc_urllc, list) else perc_urllc
        self.perc_miot = 1.0 - self.perc_embb - self.perc_urllc

        # Total UEs (ues parameter is per gNB, scenario-two has nMmWaveEnbNodes=7)
        ues_per_gnb = scenario_configuration['ues']
        self.ues_per_gnb = ues_per_gnb[0] if isinstance(ues_per_gnb, list) else ues_per_gnb
        self.num_gnbs = 7  # scenario-two always has 7 mmWave gNBs
        self.total_ues = self.ues_per_gnb * self.num_gnbs

        # State tracking
        self.last_actions = {slice_name: 0.5 for slice_name in self.slices}
        self.sla_violations_history = []
        self.total_reward = 0.0
        self.num_steps = 0

        # Calculate UE indices for each slice
        self._compute_slice_ue_indices()

        # Observation space: 12 values (4 metrics x 3 slices)
        # [throughput_embb, delay_embb, buffer_embb, split_embb,
        #  throughput_urllc, delay_urllc, buffer_urllc, split_urllc,
        #  throughput_miot, delay_miot, buffer_miot, split_miot]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )

        # Action space: 3 values (split ratio per slice, 0-1)
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

    def _compute_slice_ue_indices(self):
        """
        Compute which UEs belong to each slice based on scenario-two.cc.

        From scenario-two.cc (lines 519, 538):
        - eMBB: UEs 0 to (nUeNodes * PercUEeMBB - 1)
        - URLLC: UEs (nUeNodes * PercUEeMBB) to (nUeNodes * (PercUEeMBB + PercUEURLLC) - 1)
        - mIoT: remaining UEs
        """
        total_ues = self.total_ues

        embb_end = int(total_ues * self.perc_embb)
        urllc_end = int(total_ues * (self.perc_embb + self.perc_urllc))

        self.slice_ue_indices = {
            'embb': list(range(0, embb_end)),
            'urllc': list(range(embb_end, urllc_end)),
            'miot': list(range(urllc_end, total_ues))
        }

        if self.verbose:
            print(f"Slice UE indices: eMBB={len(self.slice_ue_indices['embb'])}, "
                  f"URLLC={len(self.slice_ue_indices['urllc'])}, "
                  f"mIoT={len(self.slice_ue_indices['miot'])}")

    def _get_ue_imsi(self, ue_idx: int) -> int:
        """
        Get the full IMSI for a UE based on its index.

        In scenario-two.cc, the basicCellId configuration determines the base IMSI.
        Assuming basicCellId=1 (default), IMSI = 1 + ue_idx.

        Args:
            ue_idx: Zero-based UE index

        Returns:
            Full IMSI (e.g., 10001 for UE 0)
        """
        # Default basicCellId is 1, and ns-3 uses 10000 as prefix
        return 10000 + ue_idx + 1

    def _get_ue_indices_for_slice(self, slice_name: str) -> List[int]:
        """Get UE indices belonging to a specific slice."""
        return self.slice_ue_indices.get(slice_name, [])

    def _read_pdcp_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Read PDCP statistics from DlE2PdcpStats.txt.

        Returns:
            Dictionary mapping UE IMSI to metrics dict
        """
        pdcp_stats = {}
        file_path = os.path.join(self.sim_path, 'DlE2PdcpStats.txt')

        if not os.path.exists(file_path):
            if self.verbose:
                print(f"Warning: PDCP stats file not found at {file_path}")
            return pdcp_stats

        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    imsi = int(row.get('IMSI', 0))
                    pdcp_stats[imsi] = {
                        'RxData': float(row.get('RxData', 0)),  # bytes
                        'AvgDelay': float(row.get('AvgDelay', 0)),  # seconds
                        'RxPackets': int(row.get('RxPackets', 0)),
                        'TxPackets': int(row.get('TxPackets', 0)),
                    }
        except Exception as e:
            if self.verbose:
                print(f"Error reading PDCP stats: {e}")

        return pdcp_stats

    def _read_rlc_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Read RLC statistics from DlE2RlcStats.txt.

        Returns:
            Dictionary mapping UE IMSI to metrics dict
        """
        rlc_stats = {}
        file_path = os.path.join(self.sim_path, 'DlE2RlcStats.txt')

        if not os.path.exists(file_path):
            if self.verbose:
                print(f"Warning: RLC stats file not found at {file_path}")
            return rlc_stats

        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    imsi = int(row.get('IMSI', 0))
                    rlc_stats[imsi] = {
                        'TxPackets': int(row.get('TxPackets', 0)),
                        'RxPackets': int(row.get('RxPackets', 0)),
                        'TxBytes': float(row.get('TxBytes', 0)),
                        'RxBytes': float(row.get('RxBytes', 0)),
                    }
        except Exception as e:
            if self.verbose:
                print(f"Error reading RLC stats: {e}")

        return rlc_stats

    def _get_slice_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregated metrics for each slice.

        Returns:
            Dictionary mapping slice names to metrics
        """
        pdcp_stats = self._read_pdcp_stats()
        rlc_stats = self._read_rlc_stats()

        slice_metrics = {}

        for slice_name in self.slices:
            ue_indices = self._get_ue_indices_for_slice(slice_name)
            ue_imsis = [self._get_ue_imsi(idx) for idx in ue_indices]

            # Filter stats for this slice's UEs
            slice_pdcp = {k: v for k, v in pdcp_stats.items() if k in ue_imsis}
            slice_rlc = {k: v for k, v in rlc_stats.items() if k in ue_imsis}

            if slice_pdcp:
                # Average throughput (convert bytes/sec to Mbps)
                throughput_sum = sum(s['RxData'] for s in slice_pdcp.values())
                avg_throughput = (throughput_sum / len(slice_pdcp) * 8) / 1e6

                # Average delay (convert seconds to ms)
                avg_delay = sum(s['AvgDelay'] for s in slice_pdcp.values()) / len(slice_pdcp) * 1000

                # Total packets for delivery rate calculation
                total_tx = sum(s['TxPackets'] for s in slice_pdcp.values())
                total_rx = sum(s['RxPackets'] for s in slice_pdcp.values())
                delivery_rate = total_rx / total_tx if total_tx > 0 else 0.0
            else:
                avg_throughput = 0.0
                avg_delay = 0.0
                delivery_rate = 0.0

            # Buffer status from RLC stats (normalized)
            if slice_rlc:
                total_buffer_packets = sum(s['TxPackets'] for s in slice_rlc.values())
                buffer_status = min(total_buffer_packets / 1e6, 1.0)  # Normalize to 0-1
            else:
                buffer_status = 0.0

            slice_metrics[slice_name] = {
                'throughput': avg_throughput,
                'delay': avg_delay,
                'buffer': buffer_status,
                'delivery_rate': delivery_rate
            }

        return slice_metrics

    def _check_sla(self, slice_name: str, metrics: Dict[str, float]) -> bool:
        """
        Check if SLA was met for a given slice.

        SLAs:
        - eMBB: throughput >= sla_embb_throughput Mbps
        - URLLC: delay <= sla_urllc_delay ms
        - mIoT: delivery_rate >= sla_miot_delivery_rate

        Args:
            slice_name: Name of the slice ('embb', 'urllc', or 'miot')
            metrics: Dictionary of metrics for the slice

        Returns:
            True if SLA is satisfied, False otherwise
        """
        if slice_name == 'embb':
            throughput_mbps = metrics.get('throughput', 0)
            return throughput_mbps >= self.sla_embb_throughput

        elif slice_name == 'urllc':
            delay_ms = metrics.get('delay', float('inf'))
            return delay_ms <= self.sla_urllc_delay

        elif slice_name == 'miot':
            delivery_rate = metrics.get('delivery_rate', 0)
            return delivery_rate >= self.sla_miot_delivery_rate

        return False

    def _compute_reward(self) -> float:
        """
        Calculate reward based on RSLAQ article adapted for PDCP splitting.

        Reward function:
        r = Σ_s (w_sla * SLA_satisfaction_s - w_opt * split_penalty_s)

        Where:
        - SLA_satisfaction: 1.0 if SLA met, 0.0 otherwise
        - split_penalty: Current split ratio (higher LTE split = penalty)

        Returns:
            Total reward for the current step
        """
        slice_metrics = self._get_slice_metrics()
        reward = 0.0
        sla_violations = []

        for slice_name in self.slices:
            sla_satisfied = self._check_sla(slice_name, slice_metrics[slice_name])

            if sla_satisfied:
                r_sla = 1.0
            else:
                r_sla = 0.0
                sla_violations.append((slice_name, slice_metrics[slice_name]))

            # Split penalty: Higher LTE split = penalty (suboptimal use of mmWave)
            current_split = self.last_actions.get(slice_name, 0.5)
            split_penalty = current_split

            # Reward for this slice (RSLAQ formula)
            r_s = self.w_sla * r_sla - self.w_opt * split_penalty
            reward += r_s

            if self.verbose:
                metrics = slice_metrics[slice_name]
                print(f"{slice_name}: SLA={'OK' if sla_satisfied else 'VIOLATED'}, "
                      f"thr={metrics['throughput']:.2f} Mbps, "
                      f"delay={metrics['delay']:.2f} ms, "
                      f"delivery={metrics['delivery_rate']:.2f}, "
                      f"split={current_split:.2f}, "
                      f"r_s={r_s:.3f}")

        # Additional penalty for multiple SLA violations
        if len(sla_violations) > 1:
            reward -= 0.5
            if self.verbose:
                print(f"Additional penalty for {len(sla_violations)} SLA violations: -0.5")

        self.sla_violations_history = sla_violations
        self.total_reward += reward
        self.num_steps += 1

        if self.verbose:
            print(f"Total reward: {reward:.3f}")

        return reward

    def _get_obs(self) -> np.ndarray:
        """
        Get current observation from the environment.

        Returns:
            Array with 12 values:
            [throughput_embb, delay_embb, buffer_embb, split_embb,
             throughput_urllc, delay_urllc, buffer_urllc, split_urllc,
             throughput_miot, delay_miot, buffer_miot, split_miot]
        """
        slice_metrics = self._get_slice_metrics()
        observations = []

        for slice_name in self.slices:
            metrics = slice_metrics.get(slice_name, {})
            current_split = self.last_actions.get(slice_name, 0.5)

            observations.extend([
                metrics.get('throughput', 0.0),
                metrics.get('delay', 0.0),
                metrics.get('buffer', 0.0),
                current_split
            ])

        return np.array(observations, dtype=np.float32)

    def _compute_action(self, action: np.ndarray) -> List[List]:
        """
        Convert agent action to ns-3 control format.

        Args:
            action: np.array([qos_embb, qos_urllc, qos_miot])
                     Values between 0.0 and 1.0

        Returns:
            List of [timestamp, ueId, percentage] entries
        """
        # Clip to ensure values are in valid range
        action = np.clip(action, 0.0, 1.0)

        cell_act_comb_lst = []
        timestamp = self.last_timestamp

        for idx, (slice_name, split_ratio) in enumerate(zip(self.slices, action)):
            ue_indices = self._get_ue_indices_for_slice(slice_name)

            # Apply the same split ratio to all UEs in the slice
            for ue_idx in ue_indices:
                ue_imsi = self._get_ue_imsi(ue_idx)
                cell_act_comb_lst.append([timestamp, ue_imsi, split_ratio])

            # Store the applied split ratio for reward calculation
            self.last_actions[slice_name] = split_ratio

        if self.verbose:
            print(f"Computed actions at timestamp {timestamp}:")
            for slice_name, split_ratio in zip(self.slices, action):
                ue_count = len(self._get_ue_indices_for_slice(slice_name))
                print(f"  {slice_name}: split={split_ratio:.2f} ({ue_count} UEs)")

        return cell_act_comb_lst

    def _init_datalake_usecase(self):
        """Initialize datalake tables specific to PDCP slicing use case."""
        # No additional tables needed for PDCP slicing use case
        pass

    def _fill_datalake_usecase(self):
        """Fill datalake with use case specific data from CSV files."""
        # PDCP slicing reads stats directly from files in _get_obs
        # No additional datalake operations needed
        pass

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action to take (split ratios for each slice)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # Add additional info about SLA violations
        info['sla_violations'] = self.sla_violations_history
        info['last_actions'] = self.last_actions.copy()
        info['total_reward'] = self.total_reward
        info['num_steps'] = self.num_steps

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset state tracking
        self.last_actions = {slice_name: 0.5 for slice_name in self.slices}
        self.sla_violations_history = []
        self.total_reward = 0.0
        self.num_steps = 0

        return obs, info
