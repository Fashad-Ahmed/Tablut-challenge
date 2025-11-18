from agent import Agent
import settings
import enviroment
import threading

class TablutTraining:
    def __init__(self, episodes: int = settings.NUM_EPISODES, max_steps: int = settings.MAX_STEPS_PER_EPISODE):
        self.env = enviroment.Environment()
        self.agent_white = Agent(is_white=True, name="AgentWhite", load_prev_path="./models/new_rewards2/white5000.pt")
        self.agent_black = Agent(is_white=False, name="AgentBlack", load_prev_path="./models/new_rewards2/black5000.pt")
        self.episodes = episodes
        self.max_steps = max_steps

    def connect_agents(self):
        exc_white = []

        def _connect_white():
            try:
                self.agent_white.connect()
            except Exception as e:
                exc_white.append(e)

        t = threading.Thread(target=_connect_white)
        t.start()
        t.join()
        if t.is_alive():
            raise TimeoutError("Timeout waiting for ehite connect() to finish")
        if exc_white:
            raise exc_white[0]

        
        exc_black = []

        def _connect_black():
            try:
                self.agent_black.connect()
            except Exception as e:
                exc_black.append(e)

        t = threading.Thread(target=_connect_black)
        t.start()
        t.join()
        if t.is_alive():
            raise TimeoutError("Timeout waiting for black connect() to finish")
        if exc_black:
            raise exc_black[0]

    def read_state(self):
        # print("White reading current state")
        self.agent_white.read_state()
        # print("Black reading current state")
        self.agent_black.read_state()
    
    def make_move(self):
        # print("Make move")
        self.agent_white.make_move()
        self.agent_black.make_move()

    def train(self):
        for episode in range(self.episodes):
            print(f"Staring episode {episode + 1}/{self.episodes}")
            self.env.run_server()

            self.connect_agents()

            step = 0
            done = False
            while step < self.max_steps or not done:
                # print(f" Step {step + 1}/{self.max_steps} for episode {episode + 1}/{self.episodes}")
                self.read_state()

                white_win = self.agent_white.win()
                black_win = self.agent_black.win()
                draw = self.agent_white.draw() or self.agent_black.draw()
                done = white_win or black_win or draw


                if step != 0:
                    self.agent_white.train()
                    self.agent_black.train()

                if done:
                    break

                self.make_move()

                step += 1
                # print(f" Step {step + 1} finished.")

            self.agent_white.close()
            self.agent_black.close()
            self.env.stop_server()

            if episode % 50 == 0:
                self.save(episode)
            print(f"Episode {episode + 1} finished.")
    
    def save(self, name=""):
        self.agent_white.save_model(name)
        self.agent_black.save_model(name)
        