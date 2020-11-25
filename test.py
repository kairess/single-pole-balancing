import pickle
import math

import neat
import gizeh as gz
import cv2

import cart_pole

# None for random values
initial_values = {
    'x': 0,
    'theta': 23 * math.pi / 180,
    'dx': 1,
    'dtheta': 1
}

simulation_seconds = 60.0
scale = 3
w, h = int(300 * scale), int(100 * scale)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('result/output.mp4', fourcc, 120, (w, h))

with open('result/winner', 'rb') as f:
    genome = pickle.load(f)

print('Loaded genome:')
print(genome)

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward'
)

net = neat.nn.FeedForwardNetwork.create(genome, config)
sim = cart_pole.CartPole(**initial_values)

while sim.t < simulation_seconds:
    inputs = sim.get_scaled_state()
    action = net.activate(inputs)

    force = cart_pole.discrete_actuator_force(action)
    sim.step(force)

    if abs(sim.x) > 3.3:
        sim.x *= -1

    cart = gz.rectangle(
        lx=25 * scale,
        ly=12.5 * scale,
        xy=(150 * scale, 80 * scale),
        fill=(0, 1, 0)
    )

    force_direction = 1 if force > 0 else -1

    force_rect = gz.rectangle(
        lx=5,
        ly=12.5 * scale,
        xy=(150 * scale - force_direction * (25 * scale) / 2, 80 * scale),
        fill=(1, 0, 0)
    )

    cart_group = gz.Group([
        cart,
        force_rect
    ])

    star = gz.star(radius=10 * scale, fill=(1, 1, 0), xy=(150 * scale, 25 * scale), angle=-math.pi / 2)

    pole = gz.rectangle(
        lx=2.5 * scale,
        ly=50 * scale,
        xy=(150 * scale, 55 * scale),
        fill=(1, 1, 0)
    )

    pole_group = gz.Group([
        pole,
        star
    ])

    # convert position to display units
    visX = sim.x * 50 * scale

    # draw background
    surface = gz.Surface(w, h, bg_color=(0, 0, 0))

    # draw cart, pole and text
    group = gz.Group([
        cart_group.translate((visX, 0)),
        pole_group.translate((visX, 0)).rotate(sim.theta, center=(150 * scale + visX, 80 * scale)),
        gz.text('Time %.2f (Fitness %.2f)' % (sim.t, genome.fitness), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 25), fontweight='bold', v_align='top', h_align='left'),
        gz.text('x: %.2f' % (sim.x,), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 50), fontweight='bold', v_align='top', h_align='left'),
        gz.text('dx: %.2f' % (sim.dx,), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 75), fontweight='bold', v_align='top', h_align='left'),
        gz.text('theta: %d' % (sim.theta * 180 / math.pi,), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 100), fontweight='bold', v_align='top', h_align='left'),
        gz.text('dtheta: %d' % (sim.dtheta * 180 / math.pi,), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 125), fontweight='bold', v_align='top', h_align='left'),
        gz.text('force: %d' % (force,), fontfamily='NanumGothic', fontsize=20, fill=(1, 0, 0), xy=(10, 150), fontweight='bold', v_align='top', h_align='left'),
    ])
    group.draw(surface)

    img = surface.get_npimage()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    out.write(img)
    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
