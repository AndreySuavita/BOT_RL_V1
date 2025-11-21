import logging
import sys
sys.path.append("../")
from utils import load_and_preprocess_data
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyReward:
    def __init__(self):
        ## Fuzzy variables
        res = 100
        price_change = ctrl.Antecedent(np.linspace(-1, 1, res), "Price Change") #array of price changes
        action = ctrl.Antecedent(np.arange(0, 3, 1), "Action") #0: Sell, 1: Hold, 2: Buy
        time_cycle = ctrl.Antecedent(np.linspace(1, 24*60, res), "time_cycle") #array of price changes

        reward = ctrl.Consequent(np.linspace(-10, 10, res), 'Reward')

        ## Membership functions for price_change
        price_change["negative"] = fuzz.gaussmf(price_change.universe, -1, 0.1)
        price_change["medium"] = fuzz.gaussmf(price_change.universe, 0, 0.1)
        price_change["positive"] = fuzz.gaussmf(price_change.universe, 1, 0.1)

        action["sell"] = fuzz.gaussmf(action.universe, 0, 0.3)
        action["hold"] = fuzz.gaussmf(action.universe, 1, 0.1)
        action["buy"] = fuzz.gaussmf(action.universe, 2, 0.3)

        time_cycle["fast"] = fuzz.gaussmf(time_cycle.universe, 1, 500)
        #time_cycle["medium"] = fuzz.gaussmf(time_cycle.universe, 700, 200)
        time_cycle["slow"] = fuzz.gaussmf(time_cycle.universe, 24*60, 500)

        reward["very_low"] = fuzz.gaussmf(reward.universe, -10, 2)
        reward["low"] = fuzz.gaussmf(reward.universe, -5, 2)
        reward["zero"] = fuzz.gaussmf(reward.universe, 0, 2)
        reward["high"] = fuzz.gaussmf(reward.universe, 5, 2)
        reward["very_high"] = fuzz.gaussmf(reward.universe, 10, 2)

        ## Fuzzy rules
        rule1 = ctrl.Rule(price_change["negative"] & time_cycle["fast"] & action['sell'], reward['low'])
        rule2 = ctrl.Rule(price_change["negative"] & time_cycle["slow"] & action['sell'], reward['very_low'])
        rule3 = ctrl.Rule(price_change["positive"] & time_cycle["fast"] & action['buy'], reward['high'])
        rule4 = ctrl.Rule(price_change["positive"] & time_cycle["slow"] & action['buy'], reward['very_high'])
        rule5 = ctrl.Rule(price_change["negative"] & time_cycle["fast"] & action['buy'], reward['very_low'])
        rule6 = ctrl.Rule(price_change["negative"] & time_cycle["slow"] & action['buy'], reward['very_low'])
        #rule7 = ctrl.Rule(price_change["medium"] & (action['sell'] | action['buy']), reward['low']) # low
        rule7 = ctrl.Rule(price_change["medium"] & action['hold'], reward['zero']) # high

        ## Fuzzy inference system
        system = ctrl.ControlSystem(rules=[rule1, rule2, rule3, rule4, rule5, rule6, rule7])
        self.sim = ctrl.ControlSystemSimulation(system)

    def __call__(self, action, price_change, time_cycle):

        self.sim.input['Price Change'] = price_change
        self.sim.input['Action'] = action
        self.sim.input['time_cycle'] = time_cycle
        try:
            self.sim.compute()
            reward = self.sim.output['Reward']
        except Exception as e:
            logging.error(f"Error in fuzzy reward computation: {e}")
            reward = 0
        return reward
    
# --- Trading Environment ---
class EnhancedTradingEnvironmentFuzzy:
    def __init__(self, data, window_size, time_cycle, scaler, binance_on=False):
        self.data = data # Normalized data
        self.window_size = window_size # Visible history 
        self.time_cycle = time_cycle # Time cycle for the data
        self.scaler = scaler # Scaler for the data
        self.binance_on = binance_on # If True, use Binance API for real-time data
        self.current_step = window_size # Current step, starts after having enough history
        self.max_steps = len(data) - 1 # Last possible step
        self.action_space = 3 # Possible actions: 0=sell, 1=hold, 2=buy
        self.state_size = window_size * self.data.shape[1] # Flattened state size (window * features) 
        self.position = 0 # 0=not invested, 1=invested (in ETH)
        self.commission = 0.001 # Commission of 0.1% per operation
        self.fuzzy_reward = FuzzyReward()
    
    def reset(self):
        if self.binance_on:
            self.data, _, _ = load_and_preprocess_data(window_size=self.window_size, time_cycle=self.time_cycle, scaler=self.scaler, binance_on=self.binance_on)
            data_flattened = self.data.flatten()  # Flatten the data
            # print(self.data.shape)
            # print(data_flattened.shape)
            # exit()
            return data_flattened # Flattened data window
        else:
            self.current_step = self.window_size
            self.position = 0
            return self._get_state()
    
    def _get_state(self):
        """
        Takes the data from the last window_size hours (e.g., 15 rows).
        flatten(): Converts the 2D matrix (15h x 8 features) into a 1D vector (for the neural network).
        """
        return self.data[self.current_step - self.window_size : self.current_step].flatten()
    
    def step(self, action):
        # If Binance is connected, return the current price
        if self.binance_on:
            current_price = self.data[self.window_size-1, 3]  # Current close price
            return {"price": current_price}
        else:
            current_price = self.data[self.current_step, 3]
            next_price = self.data[self.current_step + 1, 3] if self.current_step < self.max_steps else current_price
            
            # Safe handling of price_change calculation, division by 0
            try:
                price_change = (next_price - current_price) / current_price if current_price != 0 else 0
            except Exception as e:
                print(f"Error calculating price_change: {e}")
                price_change = 0
            
            # Validate action based on current position
            valid_action = action
            if action == 2 and self.position == 1: # Wants to buy but is already invested
                valid_action = 1 # Force hold
            elif action == 0 and self.position != 1: # Wants to sell without having a position
                valid_action = 1 # Force hold

            # if self.time_cycle == '5m':
            #     # Reward system
            #     if valid_action == 0: # Sell
            #         reward = -price_change * 1.5 # Punish selling before rises
            #         self.position = 0
            #     elif valid_action == 2: # Buy
            #         reward = price_change * 1.2 # Reward successful buys
            #         self.position = 1
            #     else: # Hold
            #         reward = 0.2 if abs(price_change) < 0.01 else -0.1 # Reward holding in sideways markets
            # elif self.time_cycle == 'hourly':
            #     # Reward system
            #     if valid_action == 0: # Sell
            #         reward = -price_change * 2.5 # Punish selling before rises
            #         self.position = 0
            #     elif valid_action == 2: # Buy
            #         reward = price_change * 2.0 # Reward successful buys
            #         self.position = 1
            #     else: # Hold
            #         reward = 0.2 if abs(price_change) < 0.01 else -0.1 # Reward holding in sideways markets
            
            # # Apply commission, action different from 1 (hold)
            # if valid_action != 1:
            #     reward -= self.commission * 2
            
            self.current_step += 1
            done = self.current_step >= self.max_steps
            next_state = self._get_state()
            """
            next_state: New state (sliding window 1 hour).

            reward: Reward/Penalty.

            done: True if the episode ended.

            info: Useful metadata (current price, valid action).
            """
            reward = self.reward_calculation(valid_action, price_change)
            
            return next_state, reward, done, {"price": current_price, "valid_action": valid_action}
        
    def reward_calculation (self, action, price_change):
        """
        Calculate the reward based on the action taken and the price change.
        """
        return self.fuzzy_reward(action, price_change, self.time_cycle)
