from .audio_tools import stft
from .audio_tools import linear_to_mel_weight_matrix
from .audio_tools import stft
from .audio_tools import iterate_invert_spectrogram
from .audio_tools import soundsc
from ..text.cleaning import text_to_sequence
from ..text.cleaning import sequence_to_text
from ..text.cleaning import cleaners
from ..text.cleaning import get_vocabulary_size

from ...core import get_logger

from scipy.io import wavfile
import numpy as np
import copy
import os

logger = get_logger()

# As originally seen in sklearn.utils.extmath
# Credit to the sklearn team
def _incremental_mean_and_var(X, last_mean=.0, last_variance=None,
                              last_sample_count=0):
    """Calculate mean update and a Youngs and Cramer variance update.
    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.
    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update
    last_mean : array-like, shape: (n_features,)
    last_variance : array-like, shape: (n_features,)
    last_sample_count : int
    Returns
    -------
    updated_mean : array, shape (n_features,)
    updated_variance : array, shape (n_features,)
        If None, only mean is computed
    updated_sample_count : int
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = X.sum(axis=0)

    new_sample_count = X.shape[0]
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = X.var(axis=0) * new_sample_count
        if last_sample_count == 0:  # Avoid division by 0
            updated_unnormalized_variance = new_unnormalized_variance
        else:
            last_over_new_count = last_sample_count / new_sample_count
            last_unnormalized_variance = last_variance * last_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance +
                new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)
        updated_variance = updated_unnormalized_variance / updated_sample_count
    return updated_mean, updated_variance, updated_sample_count


class wavfile_caching_mel_tbptt_iterator(object):
    def __init__(self, wavfile_list, txtfile_list,
                 batch_size,
                 truncation_length,
                 clean_names,
                 wav_scale = 2 ** 15,
                 window_size=512,
                 window_step=128,
                 n_mel_filters=80,
                 return_normalized=True,
                 lower_edge_hertz=125.0,
                 upper_edge_hertz=7800.0,
                 start_index=0,
                 stop_index=None,
                 cache_dir_base="/Tmp/kastner/tfbldr_cache",
                 shuffle=False, random_state=None):
         self.wavfile_list = wavfile_list
         self.wav_scale = wav_scale
         self.txtfile_list = txtfile_list
         self.batch_size = batch_size
         self.truncation_length = truncation_length
         self.random_state = random_state
         self.shuffle = shuffle
         self.cache_dir_base = cache_dir_base
         self.return_normalized = return_normalized
         self.lower_edge_hertz = lower_edge_hertz
         self.upper_edge_hertz = upper_edge_hertz
         self.clean_names = clean_names
         self.vocabulary_size = get_vocabulary_size(clean_names)
         self.window_size = window_size
         self.window_step = window_step
         self.n_mel_filters = n_mel_filters
         self.start_index = start_index
         self.stop_index = stop_index

         if shuffle and self.random_state == None:
             raise ValueError("Must pass random_state in")
         if txtfile_list is not None:
             # try to match every txt file and every wav file by name
             wv_names_and_bases = sorted([(wv.split(os.sep)[-1], str(os.sep).join(wv.split(os.sep)[:-1])) for wv in self.wavfile_list])
             tx_names_and_bases = sorted([(tx.split(os.sep)[-1], str(os.sep).join(tx.split(os.sep)[:-1])) for tx in self.txtfile_list])
             wv_i = 0
             tx_i = 0
             wv_match = []
             tx_match = []
             wv_lu = {}
             tx_lu = {}
             for txnb in tx_names_and_bases:
                 if "." in txnb[0]:
                     tx_part = ".".join(txnb[0].split(".")[:1])
                 else:
                     # support txt files with no ext
                     tx_part = txnb[0]
                 tx_lu[tx_part] = txnb[1] + os.sep + txnb[0]

             for wvnb in wv_names_and_bases:
                 wv_part = ".".join(wvnb[0].split(".")[:1])
                 wv_lu[wv_part] = wvnb[1] + os.sep + wvnb[0]

             # set of in common keys
             shared_k = sorted([k for k in wv_lu.keys() if k in tx_lu])

             for k in shared_k:
                 wv_match.append(wv_lu[k])
                 tx_match.append(tx_lu[k])
             self.wavfile_list = wv_match
             self.txtfile_list = tx_match
         self.cache = self.cache_dir_base + os.sep + "-".join(self.wavfile_list[0].split(os.sep)[1:-1])
         if not os.path.exists(self.cache):
             os.makedirs(self.cache)

         if 0 < self.start_index < 1:
             self.start_index = int(len(self.wavfile_list) * self.start_index)
         elif self.start_index >= 1:
             self.start_index = int(self.start_index)
             if self.start_index >= len(self.wavfile_list):
                 raise ValueError("start_index {} >= length of wavfile list {}".format(self.start_index, len(self.wavfile_list)))
         elif self.start_index == 0:
             self.start_index = int(self.start_index)
         else:
             raise ValueError("Invalid value for start_index : {}".format(self.start_index))

         if self.stop_index == None:
             self.stop_index = len(self.wavfile_list)
         elif 0 < self.stop_index < 1:
             self.stop_index = int(len(self.wavfile_list) * self.stop_index)
         elif self.stop_index >= 1:
             self.stop_index = int(self.stop_index)
             if self.stop_index >= len(self.wavfile_list):
                 raise ValueError("stop_index {} >= length of wavfile list {}".format(self.stop_index, len(self.wavfile_list)))
         else:
             raise ValueError("Invalid value for stop_index : {}".format(self.stop_index))

         # could match sizes here...
         self.wavfile_sizes_mbytes = [os.stat(wf).st_size // 1024 for wf in self.wavfile_list]

         if return_normalized:
             self.return_normalized = False

             # reset random seed here
             cur_random = self.random_state.get_state()

             # set up for train / test splits
             self.all_indices_ = np.arange(len(self.wavfile_list))
             self.random_state.shuffle(self.all_indices_)
             self.all_indices_ = sorted(self.all_indices_[self.start_index:self.stop_index])

             self.current_indices_ = [self.random_state.choice(self.all_indices_) for i in range(self.batch_size)]
             self.current_offset_ = [0] * self.batch_size
             self.current_read_ = [self.cache_read_wav_and_txt_features(self.wavfile_list[i], self.txtfile_list[i]) for i in self.current_indices_]
             self.to_reset_ = [0] * self.batch_size

             mean, std = self.cache_calculate_mean_and_std_normalization()
             self._mean = mean
             self._std = std

             self.random_state = np.random.RandomState()
             self.random_state.set_state(cur_random)
             self.return_normalized = True

         # set up for train / test splits
         self.all_indices_ = np.arange(len(self.wavfile_list))
         self.random_state.shuffle(self.all_indices_)
         self.all_indices_ = sorted(self.all_indices_[self.start_index:self.stop_index])

         self.current_indices_ = [self.random_state.choice(self.all_indices_) for i in range(self.batch_size)]
         self.current_offset_ = [0] * self.batch_size
         self.current_read_ = [self.cache_read_wav_and_txt_features(self.wavfile_list[i], self.txtfile_list[i]) for i in self.current_indices_]
         self.to_reset_ = [0] * self.batch_size

    def next_batch(self):
        mel_batch = np.zeros((self.truncation_length, self.batch_size, self.n_mel_filters))
        resets = np.ones((self.batch_size, 1))
        texts = []
        for bi in range(self.batch_size):
            wf, txf = self.current_read_[bi]
            if self.to_reset_[bi] == 1:
                self.to_reset_[bi] = 0
                resets[bi] = 0.
                # get a new sample
                while True:
                    self.current_indices_[bi] = self.random_state.choice(self.all_indices_)
                    self.current_offset_[bi] = 0
                    try:
                        self.current_read_[bi] = self.cache_read_wav_and_txt_features(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]])
                    except:

                        logger.info("FILE / TEXT READ ERROR {}:{}".format(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]]))
                        try:
                            self.current_read_[bi] = self.cache_read_wav_and_txt_features(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]], force_refresh=True)
                            logger.info("CORRECTED FILE / TEXT READ ERROR VIA CACHE REFRESH")
                        except:
                            logger.info("STILL FILE / TEXT READ ERROR AFTER REFRESH {}:{}".format(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]]))
                            continue
                    wf, txf = self.current_read_[bi]
                    if len(wf) > self.truncation_length:
                        break

            trunc = self.current_offset_[bi] + self.truncation_length
            if trunc >= len(wf):
                self.to_reset_[bi] = 1
            wf_sub = wf[self.current_offset_[bi]:trunc]
            self.current_offset_[bi] = trunc
            mel_batch[:len(wf_sub), bi] = wf_sub
            texts.append(txf)

        mlen = max([len(t) for t in texts])
        text_batch = np.zeros((mlen, self.batch_size, 1))
        for bi, txt in enumerate(texts):
            text_batch[:len(txt), bi, 0] = txt
        return mel_batch, text_batch, resets

    def next_masked_batch(self):
        m, t, r = self.next_batch()
        m_mask = np.ones_like(m[..., 0])
        # not ideal, in theory could also hit on 0 mels but we aren't using this for now
        # should find contiguous chunk starting from the end
        m_mask[np.sum(m, axis=-1) == 0] = 0.
        t_mask = np.zeros_like(t[..., 0])
        t_mask[t[..., 0] > 0] = 1.
        return m, m_mask, t, t_mask, r

    def cache_calculate_mean_and_std_normalization(self, n_estimate=1000):
        normpath = self._fpathmaker("norm-mean-std")
        if not os.path.exists(normpath):
            logger.info("Calculating normalization per-dim mean and std")
            for i in range(n_estimate):
                if (i % 10) == 0:
                    logger.info("Normalization batch {} of {}".format(i, n_estimate))
                m, m_mask, t, t_mask, r = self.next_masked_batch()
                m = m[m_mask > 0]
                m = m.reshape(-1, m.shape[-1])
                if i == 0:
                    normalization_mean = np.mean(m, axis=0)
                    normalization_std = np.std(m, axis=0)
                    normalization_count = len(m)
                else:
                    nmean, nstd, ncount = _incremental_mean_and_var(
                        m, normalization_mean, normalization_std,
                        normalization_count)

                    normalization_mean = nmean
                    normalization_std = nstd
                    normalization_count = ncount
            d = {}
            d["mean"] = normalization_mean
            d["std"] = normalization_std
            d["count"] = normalization_count
            np.savez(normpath, **d)
        norms = np.load(normpath)
        mean = norms["mean"]
        std = norms["std"]
        norms.close()
        return mean, std

    def calculate_log_mel_features(self, sample_rate, waveform, window_size, window_step, lower_edge_hertz, upper_edge_hertz, n_mel_filters):
        res = np.abs(stft(waveform, windowsize=window_size, step=window_step, real=False, compute_onesided=True))
        mels = linear_to_mel_weight_matrix(
            res.shape[1],
            sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=min(float(sample_rate) // 2, upper_edge_hertz),
            n_filts=n_mel_filters, dtype=np.float64)
        mel_res = np.dot(res, mels)
        log_mel_res = np.log1p(mel_res)
        return log_mel_res

    def _fpathmaker(self, fname):
        melpart = "-logmel-wsz{}-wst{}-leh{}-ueh{}-nmel{}.npz".format(self.window_size, self.window_step, int(self.lower_edge_hertz), int(self.upper_edge_hertz), self.n_mel_filters)
        if self.txtfile_list is not None:
            txtpart = "-txt-clean{}".format(str("".join(self.clean_names)))
            npzpath = self.cache + os.sep + fname + txtpart + melpart
        else:
            npzpath = self.cache + os.sep + fname + melpart
        return npzpath

    def cache_read_wav_features(self, wavpath, return_npz=False, force_refresh=False):
        fname = ".".join(wavpath.split(os.sep)[-1].split(".")[:-1])
        npzpath = self._fpathmaker(fname)
        if force_refresh or not os.path.exists(npzpath):
            sr, d = wavfile.read(wavpath)
            d = d.astype("float64")
            d = d / float(self.wav_scale)
            log_mels = self.calculate_log_mel_features(sr, d, self.window_size, self.window_step,
                                                       self.lower_edge_hertz, self.upper_edge_hertz, self.n_mel_filters)
            np.savez(npzpath, wavpath=wavpath, sample_rate=sr, log_mels=log_mels)
        npzfile = np.load(npzpath)
        log_mels = npzfile["log_mels"]
        if self.return_normalized is True:
            log_mels = (log_mels - self._mean) / self._std
        if return_npz:
            return log_mels, npzfile, npzpath
        else:
            return log_mels

    def transform_txt(self, line, txt_line=None):
        if txt_line == None:
            int_txt = text_to_sequence(line, self.clean_names)
        else:
            clean_orig_chunks = txt_line.split(" ")
            raw_chunks = line.split(" ")
            if len(raw_chunks) == len(clean_orig_chunks):
                mutated = raw_chunks
                for chunk_i in range(len(mutated)):
                    for special in "!,:?":
                        if special in clean_orig_chunks[chunk_i]:
                            if clean_orig_chunks[chunk_i][0] == special:
                                mutated[chunk_i] = special + mutated[chunk_i]
                            elif clean_orig_chunks[chunk_i][-1] == special:
                                mutated[chunk_i] = mutated[chunk_i] + special
                            #if it's in the middle we don't really know what to do... skip it
                res_txt = " ".join(mutated)
            else:
                res_txt = line
            int_txt = text_to_sequence(res_txt, self.clean_names)
        return int_txt

    def inverse_transform_txt(self, int_line):
        clean_txt = sequence_to_text(int_line, self.clean_names)
        return clean_txt

    def cache_read_txt_features(self, txtpath, npzfile=None, npzpath=None, force_refresh=False):
        if npzfile is None or "raw_txt" not in npzfile:
            with open(txtpath, "rb") as f:
                lines = f.readlines()
            raw_txt = lines[0]
            # insert commas, semicolons, punctuation, etc from original transcript...
            if "english_phone_cleaners" in self.clean_names:
                if len(lines) < 2:
                    raise ValueError("Original text not commented on second line, necessary for phone transcript")
                # skip '# '
                orig_txt = lines[1][2:]
                clean_orig_txt = cleaners.english_cleaners(orig_txt)
                int_txt = self.transform_txt(raw_txt, clean_orig_txt)
            else:
                int_txt = text_to_sequence(raw_txt, self.clean_names)

            clean_txt = sequence_to_text(int_txt, self.clean_names)

            if force_refresh or (npzfile is not None and "raw_txt" not in npzfile):
                d = {k: v for k, v in npzfile.items()}
                npzfile.close()
                d["raw_txt"] = raw_txt
                d["clean_txt"] = clean_txt
                d["int_txt"] = int_txt
                d["cleaners"] = "+".join(self.clean_names)
                np.savez(npzpath, **d)
        npzfile = np.load(npzpath)
        int_txt = npzfile["int_txt"]
        return int_txt

    def cache_read_wav_and_txt_features(self, wavpath, txtpath, force_refresh=False):
        wavfeats, npzfile, npzpath = self.cache_read_wav_features(wavpath, return_npz=True, force_refresh=force_refresh)
        txtfeats = self.cache_read_txt_features(txtpath, npzfile=npzfile, npzpath=npzpath, force_refresh=force_refresh)
        npzfile.close()
        return wavfeats, txtfeats
