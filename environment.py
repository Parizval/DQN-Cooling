import numpy as np
import math 
class Environment:

    def __init__(self,
        optimal_temperature = (18.0, 24.0),
                initial_month = 0,
                initial_ram = 10,
                initial_rate_data=60
    ):

        self.monthly_atmospheric_temperatures  = [1,5,7,10,11,20,23,24,22,10,5,1]
        self.initial_month = initial_month
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[initial_month]

        self.optimal_temperature = optimal_temperature
        self.min_temperature = -20
        self.max_temperature = 80
        self.min_ram = 10
        self.max_ram = 100
        self.max_update_ram = 5

        self.min_rate_data = 20
        self.max_rate_data = 300
        self.max_update_data = 10

        self.initial_ram = initial_ram
        self.current_ram = initial_ram
        self.initial_rate_data = initial_rate_data

        self.current_rate_data = initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_ram+ 1.25 * self.current_rate_data
        
        self.temperature_ai = self.intrinsic_temperature

        self.temperature_noai = (self.optimal_temperature[0]+ self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    def update_env(self, direction, energy_ai, month):
        
        energy_noai = 0
        if (self.temperature_noai < self.optimal_temperature[0]):
            energy_noai = self.optimal_temperature[0] - self.temperature_noai
            self.temperature_noai = self.optimal_temperature[0]
        elif (self.temperature_noai > self.optimal_temperature[1]):
            energy_noai = self.temperature_noai - self.optimal_temperature[1]
            self.temperature_noai = self.optimal_temperature[1]

        self.reward = energy_noai - energy_ai

        self.reward = math.exp(-3 * self.reward)

        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[month]

        self.current_ram += np.random.randint(-self.max_update_ram,self.max_update_ram)

        if (self.current_ram > self.max_ram):
            self.current_ram = self.max_ram
        elif (self.current_ram < self.min_ram):
            self.current_ram = self.min_ram


        self.current_rate_data += np.random.randint(-self.max_update_data,self.max_update_data)
        if (self.current_rate_data > self.max_rate_data):
            self.current_rate_data = self.max_rate_data
        elif (self.current_rate_data < self.min_rate_data):
            self.current_rate_data = self.min_rate_data

        past_intrinsic_temperature = self.intrinsic_temperature

        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_ram + 1.25 * self.current_rate_data

        delta_intrinsic_temperature = self.intrinsic_temperature - past_intrinsic_temperature


        if (direction == -1):
            delta_temperature_ai = -energy_ai
        elif (direction == 1):
            delta_temperature_ai = energy_ai

        self.temperature_ai += delta_intrinsic_temperature + delta_temperature_ai

        self.temperature_noai += delta_intrinsic_temperature


        if (self.temperature_ai < self.min_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.optimal_temperature[0] - self.temperature_ai
                self.temperature_ai = self.optimal_temperature[0]

        elif (self.temperature_ai > self.max_temperature):
            if (self.train == 1):
                self.game_over = 1
            else:
                self.total_energy_ai += self.temperature_ai - self.optimal_temperature[1]
                self.temperature_ai = self.optimal_temperature[1]
        
        self.total_energy_ai += energy_ai
        self.total_energy_noai += energy_noai

        scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/ (self.max_temperature - self.min_temperature)

        scaled_number_users = (self.current_ram - self.min_ram)/ (self.max_ram - self.min_ram)
        
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/ (self.max_rate_data - self.min_rate_data)

        next_state = np.matrix([scaled_temperature_ai,scaled_number_users,scaled_rate_data])

        return next_state, self.reward, self.game_over


    def reset(self, new_month):
        self.atmospheric_temperature = self.monthly_atmospheric_temperatures[new_month]
        self.initial_month = new_month
        self.current_ram = self.initial_ram
        self.current_rate_data = self.initial_rate_data
        self.intrinsic_temperature = self.atmospheric_temperature + 1.25 * self.current_ram + 1.25 * self.current_rate_data
        self.temperature_ai = self.intrinsic_temperature
        self.temperature_noai = (self.optimal_temperature[0] + self.optimal_temperature[1]) / 2.0
        self.total_energy_ai = 0.0
        self.total_energy_noai = 0.0
        self.reward = 0.0
        self.game_over = 0
        self.train = 1

    def observe(self):
        
        scaled_temperature_ai = (self.temperature_ai - self.min_temperature)/ (self.max_temperature - self.min_temperature)
        scaled_number_users = (self.current_ram - self.min_ram)/ (self.max_ram - self.min_ram)
        scaled_rate_data = (self.current_rate_data - self.min_rate_data)/ (self.max_rate_data - self.min_rate_data)
        current_state = np.matrix([scaled_temperature_ai,scaled_number_users,scaled_rate_data])
        
        return current_state, self.reward, self.game_over