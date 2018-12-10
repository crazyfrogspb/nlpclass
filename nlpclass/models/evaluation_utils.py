import random

import numpy as np
import sacrebleu
import torch

from nlpclass.config import model_config
from nlpclass.data.data_utils import indexesFromSentence


def beam(decoder, decoder_output, decoder_hidden, beamsize):

    indicies = []
    probs = []

    decoder_output, decoder_hidden = decoder(torch.tensor(
        decoder_output).to(torch.long).to(device), decoder_hidden)

    topv, topi = decoder_output.topk(beamsize)

    for x in range(beamsize):
        indicies.append(topi[0][x].item())
        probs.append(topv[0][x].item())

    return indicies, probs, decoder_hidden


def beams_init(beamsize, decoder, decoder_input, decoder_hidden):
    """this function creates the initial beams"""
    # create the necessary number of total beams (beams_keep)
    # populate them
    beams_keep = []

    for x in range(beamsize):
        beams_keep.append([])

    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.topk(beamsize)

    for x in range(beamsize):
        beams_keep[x].append(
            (decoder_hidden, topi[0][x].item(), topv[0][x].item()))

    return beams_keep


def beam_step(beamsize, beams_keep, decoder, step, alpha=1):
    """this function takes one step forward in the beams,
    it returns the next step the probability associated with the full beam """

    # create exploratory beams
    beams = []
    for x in range(beamsize**2):
        beams.append([])

    # store probability
    probs_vec = np.zeros(beamsize**2)

    # go through each saved beam
    for y, x in enumerate(beams_keep):

        # taking the step for that beamkeep
        indicies, probs, decoder_hidden = beam(
            decoder, x[-1][1], x[-1][0], beamsize)

        # getting the probability for the beams being explored
        beam_prob = 1
        for z in x:
            beam_prob = beam_prob * np.exp(z[2])

        # saving them
        for z, q in enumerate(probs):

            probs_vec[z + y * beamsize] = beam_prob * np.exp(q) / step**alpha

        # saving all the histories
        for t in range(beamsize):

            beams[t + y * beamsize] = x.copy()
            beams[t + y *
                  beamsize].append((decoder_hidden, indicies[t], probs[t]))

    return beams, probs_vec


def beam2word(beams_list):
    decoded_words = []
    for y in beams_list:
        # print(beams_list[y][1])
        decoded_words.append(output_lang.index2word[y])
    return decoded_words


def greedy_search(decoder_output, output_lang, ):
    decoded_words = []
    for di in range(max_length):
        topv, topi = decoder_output.topk(1)
        decoded_words.append(output_lang.index2word[topi.item()])
        decoder_input = topi.squeeze().detach()
        if topi.item() == model_config.EOS_token:
            return decoded_words, None


def create_translation(model, data, data_loader, max_length,
                       greedy=True, beam_size=3, alpha=1.0):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            total_loss, decoder_output, decoder_hidden = model(batch)

        if greedy:
            decoded_words = []
            for di in range(max_length):
                topv, topi = decoder_output.topk(1)
                decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
                if topi.item() == model_config.EOS_token:
                    return decoded_words, None
        else:
            final_sentences = []
            # to store the beams as we go through
            beams_keep = beams_init(
                beam_size, model.decoder, decoder_input, decoder_hidden)

            # the steps thereafter
            for di in range(1, max_length):
                # this function takes one step forward
                beams, probs_vec = beam_step(
                    beam_size, beams_keep, model.decoder, di, alpha)
                # this selects the best of the beams to keep
                ind = np.argpartition(probs_vec, -beam_size)[-beam_size:]
                # adjusting for deleted beams
                diff = beam_size - len(beams_keep)

                if diff > 0:
                    for z in range(diff):
                        beams_keep.append([])

                # saving the max
                for u, i in enumerate(ind):

                    beams_keep[u] = beams[i]

                # saving those beams that have EOS
                beam_copy = beams_keep.copy()
                for x in beam_copy:
                    all_words = [z[1] for z in x]
                    step_len = len(all_words)
                    if 1 in all_words:
                        associated_prob = np.prod(
                            np.array([z[2] for z in x])) / step_len**alpha
                        final_sentences.append((all_words, associated_prob))
                        beams_keep.remove(x)
                # finishing once we have enough
                if len(final_sentences) > beam_size:
                    final_probs = np.array([z[1] for z in final_sentences])
                    final_trans = beam2word(
                        final_sentences[final_probs.argmax()][0])
                    return final_trans, final_sentences


def output_to_translations(predictions, data):
    translations = []
    for row in predictions.cpu().numpy():
        decoded_words = []
        for elem in row:
            if elem not in [model_config.SOS_token, model_config.EOS_token, model_config.PAD_token]:
                decoded_words.append(data.output_lang.index2word[elem])
            if elem == model_config.EOS_token:
                break
                translations.append(' '.join(decoded_words))
        translations.append(' '.join(decoded_words))
    return translations


def bleu_eval(ref_trans, new_trans, raw_trans=True):
    # returns a bleu score
    # input lists of strings, must be of the same length!
    if raw_trans:
        return sacrebleu.raw_corpus_bleu(new_trans, [ref_trans]).score
    else:
        return sacrebleu.corpus_bleu(new_trans, [ref_trans]).score
