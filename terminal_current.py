import matplotlib.pyplot as plt
import numpy as np


def square_wave(time, amplitude_high, amplitude_low, wave_period, duty_cycle):
    normalized_time = time % wave_period #time in one period
    high_time = wave_period*duty_cycle #how long time the signal is in high state
    #print(f"norm. time: {normalized_time}, high time: {high_time}")
    
    if normalized_time < high_time:
        amplitude_at_time = amplitude_high
    else:
        amplitude_at_time = amplitude_low
    return amplitude_at_time

def square_wave2(time, amplitude_high1, time_high1, amplitude_low1, time_low1, amplitude_high2, time_high2, amplitude_low2, time_low2):
    wave_period = time_high1 + time_low1 + time_high2 + time_low2
    duty_cycle_h1 = time_high1/wave_period
    duty_cycle_l1 = (time_high1+time_low1)/wave_period
    duty_cycle_h2 = (time_high1+time_low1+time_high2)/wave_period
    
    if time<0:
        amplitude_at_time = 0
    else:
        time_in_one_period = time % wave_period

        if time_in_one_period < wave_period*duty_cycle_h1:
            amplitude_at_time = amplitude_high1
        elif (time_in_one_period >= wave_period*duty_cycle_h1) and (time_in_one_period < wave_period*duty_cycle_l1):
            amplitude_at_time = amplitude_low1
        elif (time_in_one_period >= wave_period*duty_cycle_l1) and (time_in_one_period < wave_period*duty_cycle_h2):
            amplitude_at_time = amplitude_high2
        elif (time_in_one_period >= wave_period*duty_cycle_h2):
            amplitude_at_time = amplitude_low2
        else:
            print(f"[square_wave2]WARNING unclear placement of time point. Setting current to zero")
            amplitude_at_time = 0

    return amplitude_at_time

'''
dc_pos = 14.1
dc_neg = -7.2
on_time = 5
on_zero_time = 3
off_time = 4
off_zero_time = 1

period_on = on_time + on_zero_time
period_off = off_time + off_zero_time
period_all = period_on + period_off

duty_cycle_on = on_time/period_on
duty_cycle_off = off_time/period_off
duty_cycle_all = period_on/period_all

print(f"T_on= {period_on}, duty cycle: {duty_cycle_on} \nT_off= {period_off}, duty cycle: {duty_cycle_off} \nT_all= {period_all}, duty cycle: {duty_cycle_all}")

simulation_time = np.arange(100, 153, 1)

current_at_time = [square_wave2(time, dc_pos, on_time, 0, on_zero_time, dc_neg, off_time, 0, off_zero_time) for time in simulation_time]
for time, current in zip(simulation_time, current_at_time):
    print(f"time: {time}, current: {current}")
'''






