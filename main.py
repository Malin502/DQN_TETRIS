import torch
from PlayerController import PlayerController
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)
    
if __name__ == "__main__":
    
    cont = PlayerController()
    
    mode = input("テトリスで遊ぶならplay. 学習ならtrainを入力").strip().lower()
    if mode == 'play':
        cont.play_game()
    elif mode == 'train':
        episodes = int(input("エピソード数を入力してください:"))
        cont.train_dqn(episodes)
    else:
        print("playまたはtrainを入力してください")
