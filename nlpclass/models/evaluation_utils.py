import random

import numpy as np
import torch
import sacrebleu

def beam(decoder,decoder_output,decoder_hidden,beamsize):
    
    indicies = []
    probs = []
    
    decoder_output, decoder_hidden = decoder(torch.tensor(decoder_output).to(torch.long).to(device),decoder_hidden)
    
    topv, topi = decoder_output.topk(beamsize)
    
    for x in range(beamsize):
        indicies.append(topi[0][x].item())
        probs.append(topv[0][x].item())
        
    return indicies, probs, decoder_hidden
    
def beams_init(beamsize,decoder,decoder_input,decoder_hidden):
    """this function creates the initial beams"""
    # create the necessary number of total beams (beams_keep)
    #populate them
    beams_keep = []
    
    for x in range(beamsize):
        beams_keep.append([])     
        
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.topk(beamsize)

    for x in range(beamsize):
        beams_keep[x].append((decoder_hidden, topi[0][x].item(), topv[0][x].item()))
        
    return beams_keep

def beam_step(beamsize, beams_keep, decoder, step, alpha = 1 ):
    """this function takes one step forward in the beams, 
    it returns the next step the probability associated with the full beam """
    
    #create exploratory beams
    beams = []
    for x in range(beamsize**2):
        beams.append([])
        
    #store probability
    probs_vec = np.zeros(beamsize**2)
    
    #go through each saved beam
    for y,x in enumerate(beams_keep):

        #taking the step for that beamkeep
        indicies, probs, decoder_hidden = beam(decoder,x[-1][1],x[-1][0],beamsize)

        #getting the probability for the beams being explored
        beam_prob = 1
        for z in x:
            beam_prob = beam_prob*np.exp(z[2])

        #saving them
        for z,q in enumerate(probs):

            probs_vec[z + y*beamsize] = beam_prob*np.exp(q)/step**alpha
            
        #saving all the histories
        for t in range(beamsize):

            beams[t + y*beamsize] = x.copy()
            beams[t + y*beamsize].append((decoder_hidden, indicies[t], probs[t]))
            
    return beams, probs_vec

def beam2word(beams_list):
    decoded_words = []
    for y in beams_list:
        #print(beams_list[y][1])
        decoded_words.append(output_lang.index2word[y])
    return decoded_words
    
def create_translation(encoder, decoder, sentence, max_length=MAX_LENGTH, greedy = True, beamsize = 3, alpha = 1):
    """
    Function that generate translation.
    """ 
    #this prepares for the search
    # process input sentence
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        # encode the source language
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        # decode the context vector
        decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence
        # output of this function
        
        decoder_attentions = torch.zeros(max_length, max_length)
            
        if greedy: 
            decoded_words = []
            for di in range(max_length):
            #greedy implementation
                decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
                
                topv, topi = decoder_output.topk(1)
                decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
                if topi.item() == 1: #return was <eos is produced>
                    return decoded_words, None
                    
        #beam search    
        if not greedy:
            final_sentences = []
            #to store the beams as we go through
            beams_keep = beams_init(beamsize, decoder,decoder_input,decoder_hidden)
            
            #the steps thereafter    
            for di in range(1,max_length):
                
                #this function takes one step forward
                beams, probs_vec = beam_step(beamsize, beams_keep, decoder, di, alpha)                            
                
                #this selects the best of the beams to keep
                ind = np.argpartition(probs_vec, -beamsize)[-beamsize:]
                
                #adjusting for deleted beams
                diff = beamsize - len(beams_keep)
                
                if diff > 0:
                    for z in range(diff):
                        beams_keep.append([])
                
                #saving the max
                for u,i in enumerate(ind):

                    beams_keep[u] = beams[i]
                
                #saving those beams that have EOS
                beam_copy = beams_keep.copy()
                for x in beam_copy:
  
                    all_words = [z[1] for z in x]
                    step_len = len(all_words)
                    if 1 in all_words:
                        associated_prob = np.prod(np.array([z[2] for z in x]))/step_len**alpha)
                        final_sentences.append((all_words,associated_prob))
                        beams_keep.remove(x)
                #finishing once we have enought
                if len(final_sentences) > beamsize:
                    final_probs = np.array([z[1] for z in final_sentences])
                    final_trans = beam2word(final_sentences[final_probs.argmax()][0])
                    return final_trans, final_sentences

def random_trans(encoder, decoder, n=1, max_length=200, greedy = True, beamsize = 3):  
    #for a random translation
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        if greedy:
            output_words, _ = create_translation(encoder, decoder, pair[0])
            
        else:
            output_words, translation_bank = create_translation(encoder, decoder, pair[0], max_length, greedy = False, beamsize = beamsize)
    
    output_sentence = ' '.join(output_words)
    print('<', output_sentence)
            


def one_sen_trans(encoder, decoder, sentence, view = True, max_length=200, greedy = True, beamsize = 3):  
    #to translate one specific sentence
    if greedy:
        output_words, _ = create_translation(encoder, decoder, sentence[0])
    else:
        output_words, translation_bank = create_translation(encoder, decoder, sentence[0], max_length, greedy = False, beamsize = beamsize)
        
    output_sentence = ' '.join(output_words)

    if view:    
        print('>', sentence[0])
        print('=', sentence[1])
        print('<', output_sentence)


def full_trans(encoder, decoder, pairs, max_length=200, greedy = True, beamsize = 3):  
    #to translate a corpus
    original_text = ""
    translated_text = ""
    true_translation = ""
    for x in pairs:
        original_text += x[0]
        true_translation += x[1]
        if greedy:
            output_words, _ = create_translation(encoder, decoder, x[0])
            
        else:
            output_words, translation_bank = create_translation(encoder, decoder, x[0], max_length, greedy = False, beamsize = beamsize)
        
        output_sentence = ' '.join(output_words)
        translated_text += output_sentence
    
    return original_text, true_translation, translated_text

def bleu_eval(ref_trans,new_trans,raw_trans = False):
    #returns a bleu score
    #input strings
    if raw_trans:
        return sacrebleu.raw_corpus_bleu(ref_trans,[new_trans]).score
    else:
        return sacrebleu.corpus_bleu(ref_trans,[new_trans]).score





