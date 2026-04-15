"""
Scenario dataclass and Simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Any, Dict, Tuple, Union, TYPE_CHECKING

from dynamicauc.env import AuctionEnv
from dynamicauc.policies import BiddingPolicy

import numpy as np
import pandas as pd
from dynamicauc.dataclasses import Scenario
from dynamicauc.primitives.kernel import RewardKernel

class Simulator:
    def __init__(self, scenario: Scenario, 
                 policy: Union[BiddingPolicy, Callable[[Dict[str, Any]], float], Dict[str, Union[BiddingPolicy, Callable[[Dict[str, Any]], float]]]], 
                 campaign_cfg: Dict[str, Any]):
        self.scenario = scenario
        self.policies = policy if isinstance(policy, dict) else {"default": policy}
        self.cfg = campaign_cfg

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run n_runs independent trajectories.

        Returns
        -------
        auctions_df    : one row per auction opportunity
        displays_df    : one row per won auction (display)
        conversions_df : one row per conversion event
        summary_df     : one row per run with aggregate statistics
        """
        n_runs = int(self.cfg["n_runs"])
        base_seed = int(self.cfg["seed"])

        all_auctions: List[pd.DataFrame] = []
        all_displays: List[pd.DataFrame] = []
        all_conversions: List[pd.DataFrame] = []
        all_summaries: List[pd.DataFrame] = []

        for policy_name, current_policy in self.policies.items():
            policy_auctions = []
            policy_displays = []
            policy_conversions = []
            
            for run_id in range(n_runs):
                # Seeding per run ensures identically paired environments across policies
                run_seed = base_seed + run_id
                a, d, c = self._run_single(run_id, current_policy, run_seed)
                policy_auctions.extend(a)
                policy_displays.extend(d)
                policy_conversions.extend(c)
                
            a_df = pd.DataFrame(policy_auctions)
            d_df = pd.DataFrame(policy_displays)
            c_df = pd.DataFrame(policy_conversions)
            s_df = self._summarise(a_df, c_df, n_runs)
            
            if len(self.policies) > 1 or policy_name != "default":
                a_df["policy"] = policy_name
                d_df["policy"] = policy_name
                c_df["policy"] = policy_name
                s_df["policy"] = policy_name
                
            all_auctions.append(a_df)
            all_displays.append(d_df)
            all_conversions.append(c_df)
            all_summaries.append(s_df)

        return (
            pd.concat(all_auctions, ignore_index=True),
            pd.concat(all_displays, ignore_index=True),
            pd.concat(all_conversions, ignore_index=True),
            pd.concat(all_summaries, ignore_index=True),
        )

    def _run_single(
        self,
        run_id: int,
        policy: Union[BiddingPolicy, Callable[[Dict[str, Any]], float]],
        seed: int,
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        
        env = AuctionEnv(scenario=self.scenario)
        
        # The seed sets the seed for the whole episode, ensuring pairing across policies.
        obs, _ = env.reset(seed=seed)

        auctions: List[dict] = []
        displays: List[dict] = []
        conversions: List[dict] = []
        
        done = False
        while not done:
            if hasattr(policy, "bid"):
                bid = policy.bid(obs)
            else:
                bid = policy(obs) # Allows passing a lambda or predict function directly
            
            obs, _, done, _, info = env.step(bid)

            # Collect data from the info dictionary
            auction_record = info["auction"]
            auction_record["run_id"] = run_id
            auctions.append(auction_record)

            if info["display"]:
                display_record = info["display"]
                display_record["run_id"] = run_id
                displays.append(display_record)
            
            if info["actual_conversions"]:
                for conv in info["actual_conversions"]:
                    conv["run_id"] = run_id
                conversions.extend(info["actual_conversions"])

        env.close()
        return auctions, displays, conversions

    @staticmethod
    def _summarise(
        auctions: pd.DataFrame,
        conversions: pd.DataFrame,
        n_runs: int,
    ) -> pd.DataFrame:
        records = []
        for run_id in range(n_runs):
            a = auctions[auctions.run_id == run_id] if len(auctions) else auctions
            c = conversions[conversions.run_id == run_id] if len(conversions) else conversions
            n_auc = len(a)
            n_wins = int(a.won.sum()) if n_auc else 0
            n_conv = len(c)
            n_imm = int(c.is_immediate.sum()) if n_conv else 0
            records.append(dict(
                run_id=run_id,
                n_auctions=n_auc,
                n_wins=n_wins,
                win_rate=round(n_wins / n_auc, 4) if n_auc else 0.0,
                total_spend=round(float(a.payment.sum()), 4) if n_auc else 0.0,
                n_conversions=n_conv,
                n_immediate=n_imm,
                n_delayed=n_conv - n_imm,
            ))
        return pd.DataFrame(records)
    
