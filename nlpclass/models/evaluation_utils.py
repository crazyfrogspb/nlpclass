import random

import numpy as np
import torch

from nlpclass.config import model_config
from nlpclass.models.train_utils import tensorFromSentence


def beam(decoder, decoder_output, decoder_hidden, beamsize):

    indicies = []
    probs = []

    # print(decoder_output)
    # print(decoder_hidden.shape)

    decoder_output, decoder_hidden = \
        decoder(torch.tensor(decoder_output).to(torch.long).to(device),
                decoder_hidden)

    topv, topi = decoder_output.topk(beamsize)

    for x in range(beamsize):
        indicies.append(topi[0][x].item())
        probs.append(topv[0][x].item())

    return indicies, probs, decoder_hidden


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length, greedy=True, beamsize=10):
    # process input sentence
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        # encode the source lanugage
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=model_config.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor(
            [[model_config.SOS_token]], device=model_config.device)  # SOS
        # decode the context vector
        decoder_hidden = encoder_hidden  # decoder starts from the last encoding sentence
        # output of this function
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

        # to store the beams as we go through
        beams_keep = []
        for x in range(beamsize):
            beams_keep.append([])

        for di in range(max_length):
            # greedy implementation
            if greedy:

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

                topv, topi = decoder_output.topk(1)
                decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
                if topi.item() == 1:
                    break

            else:  # beam
                beams = []
                for x in range(beamsize**2):
                    beams.append([])

                # the first step
                if di == 0:
                    probs = np.zeros(beamsize)
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(beamsize)
                    # print(topi[0][0],topv[0][0])
                    # print(topi[0][1],topv[0][1])
                    # print(topi[0][2],topv[0][2])
                    for x in range(beamsize):
                        #print(topi[0][x].item(), topv[0][x])
                        # break
                        beams_keep[x].append((decoder_hidden,
                                              topi[0][x].item(), topv[0][x].item()))
                # the steps thereafter
                else:
                    # store probability
                    probs_vec = np.zeros(beamsize**2)
                    # go through each saved beam
                    for y, x in enumerate(beams_keep):

                        # break

                        # taking the step for that beamkeep
                        indicies, probs, decoder_hidden = \
                            beam(decoder, x[-1][1], x[-1][0], beamsize)
                        # getting the probability for the beams being explored
                        beam_prob = 1
                        for z in x:
                            beam_prob = beam_prob * np.exp(z[2])

                        # saving them
                        for z, q in enumerate(probs):

                            probs_vec[z + y * beamsize] = beam_prob * np.exp(q)

                        # saving all the histories
                        for t in range(beamsize):

                            beams[t + y * beamsize] = x.copy()
                            beams[t + y * beamsize].append((decoder_hidden,
                                                            indicies[t], probs[t]))

                if di > 0:  # identifying the maxes
                    ind = np.argpartition(probs_vec, -beamsize)[-beamsize:]

                    # saving the max
                    for u, i in enumerate(ind):
                        beams_keep[u] = beams[i].copy()

                    # stopping when all are done
                    ends = []

                    for x in beams_keep:
                        # when all indicies are equal to one
                        ends.append(x[-1][1])

                    """
                    #to see if this is working
                    for x in beams_keep:
                        words_now = []
                        for y in range(len(x)):
                            words_now.append(output_lang.index2word[x[y][1]])

                        decoded_words.append(words_now)

                    for x in decoded_words:
                        print(' '.join(x))
                    #end working test
                    """

                    if len(set(ends)) == 1 and list(set(ends))[0] == 1:
                        print("end early")
                        for x in beams_keep:
                            words_now = []
                            for y in range(len(x)):
                                words_now.append(
                                    output_lang.index2word[x[y][1]])

                            decoded_words.append(words_now)

                        return decoded_words

        if not greedy:
            for x in beams_keep:
                words_now = []
                for y in range(len(x)):
                    words_now.append(output_lang.index2word[x[y][1]])

                decoded_words.append(words_now)

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, pairs, n=1, max_length=20, greedy=True, beamsize=3):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        if greedy:
            output_words = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
        else:
            output_words = evaluate(encoder, decoder, pair[0], max_length,
                                    greedy=False, beamsize=beamsize)
            # print(output_words)
            for x in output_words:
                output_sentence = ' '.join(x)
                print('<', output_sentence)
                print('')
