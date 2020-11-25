import multiprocessing as mp
import os
import pickle
import math

import neat
import gizeh as gz
import cv2

import cart_pole

# None for random values
initial_values = {
    'x': None,
    'theta': None,
    'dx': None,
    'dtheta': None
}

runs_per_net = 5
simulation_seconds = 60.0
visualize = True
use_multiprocessing = not visualize

generation = 0
scale = 3
w, h = int(300 * scale), int(100 * scale)

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = cart_pole.CartPole(**initial_values)

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness = sim.t

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def eval_genomes(genomes, config):
    global generation
    generation += 1
    best_genome = None
    best_fitness = 0

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

        if genome.fitness > best_fitness:
            best_genome = genome
            best_fitness = genome.fitness

    # visualization for best genome
    if visualize:
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        sim = cart_pole.CartPole(**initial_values)

        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

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

            star = gz.star(radius=30, fill=(1, 1, 0), xy=(150 * scale, 25 * scale), angle=-math.pi / 2)

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
                gz.text('Gen %d Time %.2f (Fitness %.2f)' % (generation, sim.t, best_genome.fitness), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 25), fontweight='bold', v_align='top', h_align='left'),
                gz.text('x: %.2f' % (sim.x,), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 50), fontweight='bold', v_align='top', h_align='left'),
                gz.text('dx: %.2f' % (sim.dx,), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 75), fontweight='bold', v_align='top', h_align='left'),
                gz.text('theta: %d' % (sim.theta * 180 / math.pi,), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 100), fontweight='bold', v_align='top', h_align='left'),
                gz.text('dtheta: %d' % (sim.dtheta * 180 / math.pi,), fontfamily='NanumGothic', fontsize=20, fill=(1, 1, 1), xy=(10, 125), fontweight='bold', v_align='top', h_align='left'),
                gz.text('force: %d' % (force,), fontfamily='NanumGothic', fontsize=20, fill=(1, 0, 0), xy=(10, 150), fontweight='bold', v_align='top', h_align='left'),
            ])
            group.draw(surface)

            img = cv2.UMat(surface.get_npimage())
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow('result', img)
            if cv2.waitKey(1) == ord('q'):
                exit()

def run():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'config-feedforward'
    )

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    if use_multiprocessing:
        pe = neat.ParallelEvaluator(mp.cpu_count(), eval_genome)
        winner = pop.run(pe.evaluate)
    else:
        winner = pop.run(eval_genomes)

    os.makedirs('result', exist_ok=True)
    with open('result/winner', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

if __name__ == '__main__':
    run()
