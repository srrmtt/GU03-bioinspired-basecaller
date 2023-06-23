import os
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler, DataLoader
from abc import abstractmethod
import numpy as np
import random
from pathlib import Path
from fast_ctc_decode import beam_search, viterbi_search, crf_greedy_search, crf_beam_search
import uuid
from tqdm import tqdm

from utils import read_metadata, time_limit, TimeoutException
from read import read_fast5
from normalization import normalize_signal_from_read_data, med_mad
from constants import CTC_BLANK, BASES_CRF, S2S_PAD, S2S_EOS, S2S_SOS, S2S_OUTPUT_CLASSES
from constants import CRF_STATE_LEN, CRF_BIAS, CRF_SCALE, CRF_BLANK_SCORE, CRF_N_BASE, BASES
from constants import STICH_ALIGN_FUNCTION, STICH_GAP_OPEN_PENALTY, STICH_GAP_EXTEND_PENALTY, RECURRENT_DECODING_DICT, MATRIX

from evaluation import alignment_accuracy, make_align_arr, elongate_cigar

from layers import CTC_CRF, BonitoLinearCRFDecoder

class BaseModelImpl(BaseModelCTC, BaseModelCRF):

    def __init__(self, decoder_type = 'ctc', *args, **kwargs):
        super(BaseModelImpl, self).__init__(*args, **kwargs)

        valid_decoder_types = ['ctc', 'crf']
        if decoder_type not in valid_decoder_types:
            raise ValueError('Given decoder_type: ' + str(decoder_type) + ' is not valid. Valid options are: ' + str(valid_decoder_types))
        self.decoder_type = decoder_type

    def decode(self, p, greedy = True, *args, **kwargs):
        """Decode the predictions
         
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            and logprobabilities
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """

        if self.decoder_type == 'ctc':
            p = p.exp().detach().cpu().numpy()
            return BaseModelCTC.decode(self, p.astype(np.float32), greedy = greedy, *args, **kwargs)
        if self.decoder_type == 'crf':
            return BaseModelCRF.decode(self, p, greedy, *args, **kwargs)
        
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        if self.decoder_type == 'ctc':
            return BaseModelCTC.calculate_loss(self, y, p)
        if self.decoder_type == 'crf':
            return BaseModelCRF.calculate_loss(self, y, p)

    def build_decoder(self, encoder_output_size, decoder_type):

        if decoder_type == 'ctc':
            decoder = nn.Sequential(nn.Linear(encoder_output_size, len(BASES)+1), nn.LogSoftmax(-1))
        elif decoder_type == 'crf':
            decoder = BonitoLinearCRFDecoder(
                insize = encoder_output_size, 
                n_base = CRF_N_BASE, 
                state_len = CRF_STATE_LEN, 
                bias=CRF_BIAS, 
                scale= CRF_SCALE, 
                blank_score= CRF_BLANK_SCORE
            )
        else:
            raise ValueError('decoder_type should be "ctc" or "crf", given: ' + str(decoder_type))
        return decoder


class BaseModelCTC(BaseModel):
    
    def __init__(self, blank = CTC_BLANK, *args, **kwargs):
        """
        Args:   
            blank (int): class index for CTC blank
        """
        super(BaseModelCTC, self).__init__(*args, **kwargs)

        self.criterions['ctc'] = nn.CTCLoss(blank = blank, zero_infinity = True).to(self.device)

    def decode(self, p, greedy = True, *args, **kwargs):
        """Decode the predictions
         
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if not isinstance(p, np.ndarray):
            p = p.cpu().numpy()

        if greedy:
            return self.decode_ctc_greedy(p, *args, **kwargs)
        else:
            return self.decode_ctc_beamsearch(p, *args, **kwargs)

    def decode_ctc_greedy(self, p, qstring = False, qscale = 1.0, qbias = 1.0, collapse_repeats = True, return_path = False, *args, **kwargs):
        """Predict the bases in a greedy approach
        Args:
            p (tensor): [len, batch, classes]
            qstring (bool): whether to return the phredq scores
            qscale (float)
            qbias (float)
        """
        
        alphabet = BASES_CRF
        decoded_predictions = list()
        
        for i in range(p.shape[1]):
            seq, path = viterbi_search(p[:, i, :], alphabet, qstring = qstring, qscale = qscale, qbias = qbias, collapse_repeats = collapse_repeats)
            if return_path:
                decoded_predictions.append((seq, path))
            else:
                decoded_predictions.append(seq)

        return decoded_predictions

    def decode_ctc_beamsearch(self, p, beam_size = 5, beam_cut_threshold = 0.1, collapse_repeats = True, *args, **kwargs):

        alphabet = BASES_CRF
        decoded_predictions = list()
        for i in range(p.shape[1]):
            seq, _ = beam_search(p[:, i, :], alphabet, beam_size = beam_size, beam_cut_threshold = beam_cut_threshold, collapse_repeats = collapse_repeats)
            decoded_predictions.append(seq)

        return decoded_predictions
            
    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss = self.calculate_ctc_loss(y, p)
        losses = {'loss.global': loss.item(), 'loss.ctc': loss.item()}

        return loss, losses

    def calculate_ctc_loss(self, y, p):
        """Calculates the ctc loss
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
        """
        
        y_len = torch.sum(y != CTC_BLANK, axis = 1).to(self.device)
        p_len = torch.full((p.shape[1], ), p.shape[0]).to(self.device)
        
        loss = self.criterions["ctc"](p, y, p_len, y_len)
        
        return loss

class BaseModelCRF(BaseModel):
    
    def __init__(self, state_len = 4, alphabet = BASES_CRF, *args, **kwargs):
        """
        Args:
            state_len (int): k-mer length for the states
            alphabet (str): bases available for states, defaults 'NACGT'
        """
        super(BaseModelCRF, self).__init__(*args, **kwargs)

        self.alphabet = alphabet
        self.state_len = alphabet
        self.seqdist = CTC_CRF(state_len = state_len, alphabet = alphabet)
        self.criterions = {'crf': self.seqdist.ctc_loss}

        
    def decode(self, p, greedy = True, *args, **kwargs):
        """Decode the predictions
        
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if greedy:
            return self.decode_crf_greedy(p, *args, **kwargs)
        else:
            return self.decode_crf_beamsearch(p, *args, **kwargs)

    def compute_scores(self, probs, use_fastctc = False):
        """
        Args:
            probs (cuda tensor): [length, batch, channels]
            use_fastctc (bool)
        """
        if use_fastctc:
            scores = probs.cuda().to(torch.float32)
            betas = self.seqdist.backward_scores(scores.to(torch.float32))
            trans, init = self.seqdist.compute_transition_probs(scores, betas)
            trans = trans.to(torch.float32).transpose(0, 1)
            init = init.to(torch.float32).unsqueeze(1)
            return (trans, init)
        else:
            scores = self.seqdist.posteriors(probs.cuda().to(torch.float32)) + 1e-8
            tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
            return tracebacks

    def _decode_crf_greedy_fastctc(self, tracebacks, init, qstring, qscale, qbias, return_path):
        """
        Args:
            tracebacks (np.array): [len, states, bases]
            init (np.array): [states]
            qstring (bool)
            qscale (float)
            qbias (float)
            return_path (bool)
        """

        seq, path = crf_greedy_search(
            network_output = tracebacks, 
            init_state = init, 
            alphabet = BASES_CRF, 
            qstring = qstring, 
            qscale = qscale, 
            qbias = qbias
        )
        if return_path:
            return seq, path
        else:
            return seq
    
    def decode_crf_greedy(self, probs, use_fastctc = False, qstring = False, qscale = 1.0, qbias = 1.0, return_path = False, *args, **kwargs):
        """Predict the sequences using a greedy approach
        
        Args:
            probs (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """

        if use_fastctc:
            tracebacks, init = self.compute_scores(probs, use_fastctc)
            return self._decode_crf_greedy_fastctc(tracebacks, init, qstring, qscale, qbias, return_path)
        
        else:
            return [self.seqdist.path_to_str(y) for y in self.compute_scores(probs, use_fastctc).cpu().numpy()]

    def _decode_crf_beamsearch_fastctc(self, tracebacks, init, beam_size, beam_cut_threshold, return_path):
        """
        Args
            tracebacks (np.array): [len, states, bases]
            init (np.array): [states]
            beam_size (int)
            beam_cut_threshold (float)
            return_path (bool)
        """
        seq, path = crf_beam_search(
            network_output = tracebacks, 
            init_state = init, 
            alphabet = BASES_CRF, 
            beam_size = beam_size,
            beam_cut_threshold = beam_cut_threshold
        )
        if return_path:
            return seq, path
        else:
            return seq

    def decode_crf_beamsearch(self, probs, beam_size = 5, beam_cut_threshold = 0.1, return_path = False, *args, **kwargs):
        """Predict the sequences using a beam search
        
        Args:
            probs (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """

        tracebacks, init = self.compute_scores(probs, use_fastctc = True)
        return self._decode_crf_beamsearch_fastctc(tracebacks, init, beam_size, beam_cut_threshold, return_path)

    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss = self.calculate_crf_loss(y, p)
        losses = {'loss.global': loss.item(), 'loss.crf': loss.item()}

        return loss, losses

    def calculate_crf_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """

        y_len = torch.sum(y != CTC_BLANK, axis = 1).to(self.device)
        loss = self.criterions['crf'](scores = p, 
                                      targets = y, 
                                      target_lengths = y_len, 
                                      loss_clip = 10, 
                                      reduction='mean', 
                                      normalise_scores=True)
        return loss


class BaseBasecaller():

    def __init__(self, dataset, model, batch_size, output_file, n_cores = 4, chunksize = 2000, overlap = 200, stride = None, beam_size = 1, beam_threshold = 0.1):

        assert isinstance(dataset, BaseFast5Dataset)

        self.dataset = DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 2)
        self.model = model
        self.batch_size = batch_size
        self.output_file = output_file
        self.n_cores = n_cores
        self.chunksize = chunksize
        self.overlap = overlap
        if stride is None:
            self.stride = self.model.cnn_stride
        else:
            self.stride = stride
        self.beam_size = beam_size
        self.beam_threshold = beam_threshold

    def stich(self, method, *args, **kwargs):
        """
        Stitch chunks together with a given overlap
        
        Args:
            chunks (tensor): predictions with shape [samples, length, classes]
        """

        if method == 'stride':
            return self.stich_by_stride(*args, **kwargs)
        elif method == 'alignment':
            return self.stich_by_alignment(*args, **kwargs)
        else:
            raise NotImplementedError()

    def basecall(self, verbose = True):
        raise NotImplementedError()
    
    def stitch_by_stride(self, chunks, chunksize, overlap, length, stride, reverse=False):
        """
        Stitch chunks together with a given overlap
        
        This works by calculating what the overlap should be between two outputed
        chunks from the network based on the stride and overlap of the inital chunks.
        The overlap section is divided in half and the outer parts of the overlap
        are discarded and the chunks are concatenated. There is no alignment.
        
        Chunk1: AAAAAAAAAAAAAABBBBBCCCCC
        Chunk2:               DDDDDEEEEEFFFFFFFFFFFFFF
        Result: AAAAAAAAAAAAAABBBBBEEEEEFFFFFFFFFFFFFF
        
        Args:
            chunks (tensor): predictions with shape [samples, length, *]
            chunk_size (int): initial size of the chunks
            overlap (int): initial overlap of the chunks
            length (int): original length of the signal
            stride (int): stride of the model
            reverse (bool): if the chunks are in reverse order
            
        Copied from https://github.com/nanoporetech/bonito
        """

        if isinstance(chunks, np.ndarray):
            chunks = torch.from_numpy(chunks)

        if chunks.shape[0] == 1: return chunks.squeeze(0)

        semi_overlap = overlap // 2
        start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
        stub = (length - overlap) % (chunksize - overlap)
        first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

        if reverse:
            chunks = list(chunks)
            return torch.cat([
                chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
            ])
        else:
            return torch.cat([
                chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
            ])

    def stich_by_alignment(self, preds, qscores_list, num_patch_bases = 10):

        consensus = list()
        phredq_consensus = list()
        for i in range(0, len(preds) - 2, 2):
            
            if i == 0:
                ref1 = preds[i]
                ref1_phredq = qscores_list[i]

            ref2 = preds[i+2]
            ref2_phredq = qscores_list[i+2]
            que = preds[i+1]
            que_phredq = qscores_list[i+1]

            alignment = STICH_ALIGN_FUNCTION(que, ref1+ref2, open = STICH_GAP_OPEN_PENALTY, extend = STICH_GAP_EXTEND_PENALTY, matrix = MATRIX)

            decoded_cigar = alignment.cigar.decode.decode()
            long_cigar, _, _ = elongate_cigar(decoded_cigar)
            align_arr = make_align_arr(long_cigar, ref1+ref2, que, phredq = que_phredq, phredq_ref = ref1_phredq+ref2_phredq)


            n_gaps = 0
            st_first_segment = len(ref1) - num_patch_bases
            while True:
                n_gaps_new = np.sum(align_arr[0][:st_first_segment] == '-')
                if n_gaps_new > n_gaps:
                    st_first_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            n_gaps = 0
            nd_first_segment = st_first_segment + num_patch_bases
            while True:
                n_gaps_new = np.sum(align_arr[0][st_first_segment:nd_first_segment] == '-')
                if n_gaps_new > n_gaps:
                    nd_first_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            st_second_segment = nd_first_segment
            nd_second_segment = st_second_segment + num_patch_bases

            n_gaps = 0
            while True:
                n_gaps_new = np.sum(align_arr[0][st_second_segment:nd_second_segment] == '-')
                if n_gaps_new > n_gaps:
                    nd_second_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            segment1_patch = "".join(align_arr[2][st_first_segment:nd_first_segment].tolist()).replace('-', '')
            segment2_patch = "".join(align_arr[2][st_second_segment:nd_second_segment].tolist()).replace('-', '')
            segment1_patch_phredq = "".join(align_arr[3][st_first_segment:nd_first_segment].tolist()).replace(' ', '')
            segment2_patch_phredq = "".join(align_arr[3][st_second_segment:nd_second_segment].tolist()).replace(' ', '')

            new_ref1 = ref1[:-num_patch_bases] + segment1_patch
            new_ref1_phredq = ref1_phredq[:-num_patch_bases] + segment1_patch_phredq
            ref1 = segment2_patch + ref2[num_patch_bases:] 
            ref1_phredq = segment2_patch_phredq + ref2_phredq[num_patch_bases:] 
            assert len(new_ref1) == len(new_ref1_phredq)

            consensus.append(new_ref1)
            phredq_consensus.append(new_ref1_phredq)

        return "".join(consensus), "".join(phredq_consensus), '+'  

    def stich_by_alignment2(self, preds, qscores_list, chunk_size, chunk_overlap):
        """Stich basecalled sequences via alignment

        Works the same as stich_by_stride, but instead on overlaping the windows
        based on the stride. We overlap the basecalled sequences based on 
        alignment.

        Args:
            preds(list): list of predicted sequences as strings
            qscores(list): list of phredq scores as chars
            chunk_size (int): length of the raw data window
            chunk_overlap (int): length of the overlap between windows
        """
        pre_st = 0
        consensus = list()
        phredq_consensus = list()
        unmasked_fraction = round(chunk_overlap/chunk_size, 1)
        min_length_align = int(chunk_overlap*unmasked_fraction/20)
        difficult_alignments = 0

        for i in range(len(preds)-1):

            que = preds[i+1]
            ref = preds[i]
            ref_phredq = qscores_list[i]
            que_phredq = qscores_list[i+1]
            right = len(ref) - int(len(ref)*unmasked_fraction)
            left = int(len(ref)*unmasked_fraction)

            cut_ref = ref[-left:]
            cut_que = que[:int(len(que)*unmasked_fraction)]
            cut_ref_phredq = ref_phredq[-left:]
            cut_que_phredq = que_phredq[:int(len(que)*unmasked_fraction)]

            if len(cut_ref) <=  min_length_align or len(cut_que) <= min_length_align:
                pre_st = 0
                consensus.append(ref)
                phredq_consensus.append(ref_phredq)
                difficult_alignments += 1
                continue

            alignment = STICH_ALIGN_FUNCTION(cut_que, cut_ref, open = STICH_GAP_OPEN_PENALTY, extend = STICH_GAP_EXTEND_PENALTY, matrix = MATRIX)

            decoded_cigar = alignment.cigar.decode.decode()
            long_cigar, _, _ = elongate_cigar(decoded_cigar)
            align_arr = make_align_arr(long_cigar, cut_ref, cut_que, cut_que_phredq, cut_ref_phredq)
            matches = np.where(align_arr[1] == '|')[0]
            missmatches = np.where(align_arr[1] == '.')[0]

            st = min(np.concatenate([matches, missmatches]))
            nd = max(np.concatenate([matches, missmatches])) + 1
            mid = int((nd - st) /2)
            
            mini1 = align_arr[0][:st + mid]
            mini2 = align_arr[2][(st + mid):nd]
            mini1 = "".join(mini1.tolist()).replace('-', '')
            mini2 = "".join(mini2.tolist()).replace('-', '')

            mini1_phredq = align_arr[4][:st + mid]
            mini2_phredq = align_arr[3][(st + mid):nd]
            mini1_phredq = "".join(mini1_phredq.tolist()).replace(' ', '')
            mini2_phredq = "".join(mini2_phredq.tolist()).replace(' ', '')

            pre_st = nd - st

            consensus_seq = ref[pre_st:right]+mini1+mini2
            consensus_phredq = ref_phredq[pre_st:right]+mini1_phredq+mini2_phredq
            assert len(consensus_seq) == len(consensus_phredq)
            consensus.append(consensus_seq)
            phredq_consensus.append(consensus_phredq)

        if difficult_alignments > 0:
            direction = 'difficult_' + str(difficult_alignments)
        else:
            direction = '+'

        return "".join(consensus), "".join(phredq_consensus), direction 