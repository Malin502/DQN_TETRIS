from PlayerController import PlayerController
    
    
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
