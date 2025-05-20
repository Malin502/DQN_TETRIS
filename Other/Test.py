import argparse
import torch
import pygame
from Tetris import Tetris
from Agent import DeepQNetwork

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--model_path", type=str, default="MyModel.pth", help="Path to the trained model")

    args = parser.parse_args()
    return args

def test(opt):
    model = DeepQNetwork()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("MyModel_sugoi.pth"))
    else:
        model.load_state_dict(torch.load("MyModel.pth", map_location=torch.device('cpu')))
    
    model.eval()
    env = Tetris(height=opt.height, width=opt.width, block_size=opt.block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()

    pygame.init()
    screen = pygame.display.set_mode((opt.width * opt.block_size + 200, opt.height * opt.block_size))
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        next_steps, lines_cleared = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action)
        env.render()

        if done:
            env.reset()
            continue

        pygame.display.flip()
        clock.tick(opt.fps)

    pygame.quit()

if __name__ == "__main__":
    opt = get_args()
    test(opt)
