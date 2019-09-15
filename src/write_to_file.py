#!/usr/bin/env python3

'''
Run this module to perform Value Iteration and Q-Learning on "L-track" and
"R-track" with two crash scenarios. Training processes and test resluts are
written to files. Figures of learning curves are also created
'''

from value_iteration import ValueIteration
from q_learning import QLearning
import matplotlib.pyplot as plt

def main():
    tracks = ['L-track.txt', 'R-track.txt']
    models = [ValueIteration, QLearning]
    for track in tracks:
        out_file = open(track[:7] + '-output.txt', 'w')
        out_file.write('Perform reinforcement learning on ' + track + '\n')
        for model in models:
            if model == ValueIteration:
                iterations = 1000
                out_file.write('\n')
                out_file.write(' --------------- \n')
                out_file.write('|Value Iteration|\n')
                out_file.write(' --------------- \n')
                for crash_type in ([0, 1]):
                    if crash_type == 0:
                        crash = 'Mild Crash'
                    else:
                        crash = 'Harsh Crash'
                    r_learning = model(track)
                    learning_curve = r_learning.train(iterations, crash_type)
                    out_file.write('\n')
                    out_file.write(crash + '\n')
                    out_file.write('----------\n')
                    out_file.write('\n')
                    out_file.write('Training process:\n')
                    out_file.write('\n')
                    out_file.write('              Max value difference\n')
                    for i in range(len(learning_curve)):
                        out_file.write('Iteration #' + str(i + 1) + ': ' + str(learning_curve[i]) + '\n')
                    out_file.write('\n')
                    out_file.write('--> Values converged!\n')

                    plt.close()
                    plt.plot(learning_curve)
                    plt.title(track[:7] + 'Value Iteration - ' + crash)
                    plt.xlabel('Iteration #')
                    plt.ylabel('Maximum value difference')
                    if crash_type == 0:
                        plt.savefig(track[:7] + '-value-iteration-mild-crash-learning-curve.png')
                    else:
                        plt.savefig(track[:7] + '-value-iteration-harsh-crash-learning-curve.png')


                    out_file.write('\n')
                    out_file.write('Testing:\n')
                    average_steps = 0
                    for i in range(10):
                        steps = r_learning.test(crash_type, True)
                        average_steps += steps
                        out_file.write('\n')
                        out_file.write('Test run #' + str(i + 1) + '\n')
                        r_learning.write_all(out_file)
                    average_steps /= 10
                    out_file.write('\n')
                    out_file.write('-----------------------\n')
                    out_file.write('Average steps taken: ' + str(average_steps + 1) + '\n')
                    out_file.write('\n')

            else:
                iterations = 10000
                max_steps = 1000
                if track == 'R-track.txt':
                    max_steps = 10000
                out_file.write('\n')
                out_file.write(' ---------- \n')
                out_file.write('|Q-Learning|\n')
                out_file.write(' ---------- \n')
                for crash_type in ([0, 1]):
                    if crash_type == 0:
                        crash = 'Mild Crash'
                    else:
                        crash = 'Harsh Crash'
                    r_learning = model(track)
                    learning_curve = r_learning.train(iterations, crash_type, max_steps)
                    out_file.write('\n')
                    out_file.write(crash + '\n')
                    out_file.write('----------\n')
                    out_file.write('\n')
                    out_file.write('Training process:\n')
                    out_file.write('\n')
                    out_file.write('              Steps taken before finding the finish line\n')
                    for i in range(len(learning_curve)):
                        out_file.write('Iteration #' + str(i + 1) + ': ' + str(learning_curve[i]) + '\n')
                    out_file.write('\n')
                    out_file.write('--> Training completed!\n')

                    plt.close()
                    plt.plot(learning_curve)
                    plt.title(track[:7] + 'Q-Learning - ' + crash)
                    plt.xlabel('Iteration #')
                    plt.ylabel('Steps taken')
                    if crash_type == 0:
                        plt.savefig(track[:7] + '-Q-learning-mild-crash-learning-curve.png')
                    else:
                        plt.savefig(track[:7] + '-Q-learning-harsh-crash-learning-curve.png')

                    out_file.write('\n')
                    out_file.write('Testing:\n')
                    average_steps = 0
                    for i in range(10):
                        steps = r_learning.test(crash_type, True)
                        average_steps += steps
                        out_file.write('\n')
                        out_file.write('Test run #' + str(i + 1) + '\n')
                        r_learning.write_all(out_file)
                    average_steps /= 10
                    out_file.write('\n')
                    out_file.write('-----------------------\n')
                    out_file.write('Average steps taken: ' + str(average_steps + 1) + '\n')
                    out_file.write('\n')

        out_file.close()

if __name__ == '__main__':
    main()
