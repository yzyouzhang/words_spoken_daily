import librosa
import librosa.display
from librosa.output import write_wav
import numpy as np
import torch
import torch.nn.functional as F
import glob
from tqdm import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from IPython.display import HTML
import soundfile as sf
import os
import time
import warnings
import argparse
from scipy.signal import medfilt
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sampling_rate = 16000
sr = 16000
seg_len = 1
hop = 0.02


def load_and_display(filename, st, ed, note):
    speakAD, sr = librosa.load(filename, sr=sampling_rate, offset=st, duration=ed - st)
    speakAD = librosa.util.normalize(speakAD)
    return speakAD


def get_SVscore(embed_model_path, speakAD_noisy_test, speakAD_noisy_enrollment):
    emb_model = load_embedder(model_name=embed_model_path)
    enrollment = librosa.util.normalize(speakAD_noisy_enrollment)
    mix = librosa.util.normalize(speakAD_noisy_test)
    scores_ori = speaker_vad(emb_model, enrollment, mix, seg_len=seg_len, hop=hop)
    return scores_ori


def load_embedder(model_name='model'):
    # import sys
    # sys.path.append(os.getcwd())
    # sys.path.append('/home/neil/e2e-sv')
    model_dir = model_name
    models = torch.load('{}/embedding_models.pt'.format(model_dir))
    embed_model = models[-1].cuda()
    embed_model.eval()
    return embed_model


def speaker_vad(embedding_model, enrollment, mix, seg_len=4.0, hop=1.0):
    enrollment = librosa.util.normalize(enrollment)
    enrollment = torch.from_numpy(enrollment).float()
    enroll_embedding = embedding_model(enrollment.unsqueeze(0).unsqueeze(0).cuda())
    scores = []
    seg_len = int(sampling_rate * seg_len)
    hop = int(sampling_rate * hop)
    num_segs = int(len(mix) / hop)
    for i in tqdm(range(num_segs)):
        if i * hop < int(seg_len / 2):
            #             left_padding = np.zeros(int(seg_len/2) - i*hop)
            left_padding = mix[:int(seg_len / 2) - i * hop]  # better results
            segment = np.concatenate((left_padding, mix[0:int(seg_len / 2) + i * hop]), axis=0)
        elif (i * hop + int(seg_len / 2)) > len(mix):
            #             right_padding = np.zeros(i*hop + int(seg_len/2) - len(mix))
            right_padding = mix[i * hop + int(seg_len / 2) - len(mix):]  # better results
            segment = np.concatenate((mix[i * hop - int(seg_len / 2):], right_padding), axis=0)
        else:
            segment = mix[i * hop - int(seg_len / 2): int(seg_len / 2) + i * hop]
        segment = librosa.util.normalize(segment)
        segment = torch.from_numpy(segment).float()
        seg_embedding = embedding_model(segment.unsqueeze(0).unsqueeze(0).cuda())
        scores.append(F.cosine_similarity(enroll_embedding, seg_embedding).item())

    return scores


def get_seg_lst(score, threshold, seg_len, hop):
    st_ed_lst = []
    seg_len = int(sampling_rate * seg_len)
    filt_score_nr = medfilt(score, int(2 / hop + 1))
    hop = int(sampling_rate * hop)
    for i in range(len(filt_score_nr)):
        start = i * hop - int(seg_len / 4)
        end = i * hop + int(seg_len / 4)
        score_unit = filt_score_nr[i]
        if score_unit > threshold:
            if len(st_ed_lst) > 0:
                st_prev, ed_prev = st_ed_lst[-1]
                if start < ed_prev:
                    st_ed_lst[-1] = st_prev, end
                else:
                    st_ed_lst.append([start, end])
            else:
                st_ed_lst.append([start, end])
    return st_ed_lst


def extract_target_speaker_direct(filename, enrollment, threshold, args):
    audio, sr = librosa.load(filename, sr=sampling_rate)
    basename = os.path.basename(filename)[:-4]
    segment_length = 3200000
    num_segs = len(audio) // segment_length + 1
    extracted = np.zeros(1)
    for j in range(num_segs):
        start = j * segment_length
        segment = audio[start:start + segment_length]
        start = start / sr
        score_nr = get_SVscore('./model', segment, enrollment)
        seg_lst = get_seg_lst(score_nr, threshold, seg_len, hop)
        # calculate time stamp
        annotation_file = os.path.join(args.out_folder, basename + "_extracted.txt")
        for seg in seg_lst:
            f = open(annotation_file, "a")
            f.write("%.2f\t%.2f\n" % (seg[0] / sr + start, seg[1] / sr + start))
            f.close()
            extracted = np.concatenate((extracted, segment[seg[0]:seg[1]]), axis=0)
    write_wav(os.path.join(args.out_folder, basename + "_extracted.wav"), extracted, sr=sr)


def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Path prepare
    parser.add_argument("-i", "--input_folder", type=str, help="input folder",
                        required=True, default="/data/neil/speak-ad/enrollment_passages")
    parser.add_argument("-o", "--out_folder", type=str, help="output folder",
                        required=True, default='/data/neil/speak-ad/extracted/')
    parser.add_argument("-e", "--enroll", type=str, help="enroll wav path",
                        required=True, default='/data/neil/speak-ad/enrollment_passages/AD_01_6_24_2019_enrollment_passage.wav')

    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument("-t", "--threshold", type=float, help="threshold of cosine similarity for determine the speaker",
                        required=True, default=0.5)
    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


if __name__ == "__main__":
    args = initParams()
    print(args)

    for filename in glob.glob(os.path.join(args.input_folder, "*.WAV")):
        enrollment = load_and_display(args.enroll, 200, 250, "enrollment")
        extract_target_speaker_direct(filename, enrollment, args.threshold, args)




