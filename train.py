from agent.agent import Agent
from functions import *
import sys
import numpy as np
import matplotlib.pyplot as plt



from keras.callbacks import TensorBoard, EarlyStopping

try:
	if len(sys.argv) != 4:
		print ("Usage: python train.py [stock] [window] [episodes]")
		exit()

	stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

	agent = Agent(window_size)
	data = getStockDataVec(stock_name)
	

	l = len(data) - 1
	batch_size = 32

	for e in range(episode_count + 1):
		print ("Episode " + str(e) + "/" + str(episode_count))
		state = getState(data, 0, window_size + 1)

		total_profit = 0
		agent.inventory = []
		hold_try_profit = []
		

		for t in range(l):
			action = agent.act(state)

			# sit
			next_state = getState(data, t + 1, window_size + 1)
			reward = 0
			if t == l-1:
				clear_position = len(agent.inventory)
				reward = clear_position*data[t] - sum(agent.inventory)
				agent.rewardmemory.append(reward) #

				total_profit += reward
				agent.total_profit.append(total_profit)
				if e == episode_count:
					agent.final_try_profit.append(total_profit)

				if e == 0:
					agent.first_try_profit.append(total_profit)
				
				print ("Clear position. Sell: " +str(clear_position)+" stocks at "+formatPrice(data[t]) + " | Profit: " + formatPrice(reward))
				continue

			if action == 1: # buy
				agent.inventory.append(data[t])

				print ("Buy: " + formatPrice(data[t]))

			elif action == 2 and len(agent.inventory) > 0: # sell
				
				bought_price = agent.inventory.pop(0)
				reward = data[t] - bought_price# consider negative values

				agent.rewardmemory.append(reward) #


				total_profit += data[t] - bought_price
				agent.total_profit.append(total_profit)

				if e == episode_count:
					agent.final_try_profit.append(total_profit)

				if e == 0:
					agent.first_try_profit.append(total_profit)


					

				print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

			done = True if t == l - 1 else False
			agent.memory.append((state, action, reward, next_state, done))
			state = next_state

			if done:
				print ("--------------------------------")
				print ("Total Profit: " + formatPrice(total_profit))
				print ("--------------------------------")

			if len(agent.memory) > batch_size:
				agent.expReplay(batch_size)

		agent.episode_memory.append(total_profit)



		if e % 10 == 0:
			agent.model.save("models/model_ep" + str(e))

	
	t0 = np.arange(1, len(hold_try_profit)+1, 1)
	t1 = np.arange(1, len(agent.rewardmemory)+1, 1)
	t2 = np.arange(1, len(agent.total_profit)+1, 1)
	t3 = np.arange(1, len(agent.first_try_profit)+1, 1)
	t4 = np.arange(1, len(agent.final_try_profit)+1, 1)
	t5 = np.arange(1, len(agent.episode_memory)+1, 1)

	plt.figure(figsize=(20,20))
	plt.subplot(511)
	plt.plot(t1, agent.rewardmemory, '-')
	plt.title('Agent reward')

	plt.subplot(512)
	plt.plot(t2, agent.total_profit, '-')
	plt.title('Agent total_profit')

	#plt.subplot(614)
	#plt.plot(t0,hold_try_profit, '-')
	#plt.title('Agent First episode profit')

	plt.subplot(513)
	plt.plot(t3,agent.first_try_profit, '-')
	plt.title('Agent First episode profit')

	plt.subplot(514)
	plt.plot(t4,agent.final_try_profit, '-')
	plt.title('Agent Last episode profit')

	plt.subplot(515)
	plt.plot(t5,agent.episode_memory, '-')
	plt.title('Agent profit for each episodes')


	plt.savefig('performance.png')

	
except Exception as e:
	print("Error occured: {0}".format(e))
finally:
	exit()