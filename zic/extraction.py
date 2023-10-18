import numpy as np
import scipy.io
from scipy import signal
data = np.load('H:/data.npy')
label = np.load('H:/label.npy')


def filter_filter(input_dataa, low, high, sr):
    sos = scipy.signal.cheby2(4, 20, [low * 2 / sr, high * 2 / sr], btype='bandpass', output='sos', fs=sr)
    filtered = signal.sosfilt(sos, input_dataa)
    return filtered


def csp_filter(csp_data, csp_label):
    trialNo = len(csp_label)
    classNo = len(np.unique(csp_label))
    channelNo = csp_data.shape[1]
    projM = np.zeros((classNo, channelNo, channelNo))
    for e in range(classNo):
        N_a = sum(csp_label == e)
        N_b = trialNo - N_a
        R_a = np.zeros((channelNo, channelNo))
        R_b = np.zeros((channelNo, channelNo))
        for f in range(trialNo):
            csp_data_f = csp_data[f, :, :]
            csp_data_transpose = csp_data_f.transpose()
            R = np.dot(csp_data_f, csp_data_transpose)
            R = R / R.trace()
            if csp_label[f] == e:
                R_a = R_a + R
            else:
                R_b = R_b + R
        R_a = R_a / N_a
        R_b = R_b / N_b
        R3 = R_a + R_b
        U, S, V = np.linalg.svd(R3)
        K = np.dot(U, V)
        S = S ** (-0.5)
        S = np.array(S)
        S = abs(np.sort(-S))
        SS = np.zeros((channelNo, channelNo))
        for ss in range(channelNo):
            SS[ss, ss] = S[ss]
        P = np.dot(SS, V)
        transformedCov1 = np.dot(np.dot(P, R_a), P.T)
        D1, S1, V1 = np.linalg.svd(transformedCov1)
        projM[e, :, :] = np.dot(V1, P)
    return projM


def csp_feature(project, input_feature, selectNum):
    input_feature = input_feature.transpose()
    claN = project.shape[0]
    chaN = input_feature.shape[1]
    feature = np.zeros((1, 2 * selectNum * claN))
    for g in range(claN):
        Z = np.dot(input_feature, project[g, :, :])
        for h in range(selectNum):
            column1 = np.var(Z[:, h])
            column2 = np.var(Z[:, chaN - h - 1])
            feature[0, h + g * selectNum * 2] = column1
            feature[0, 2 * selectNum + g * selectNum * 2 - 1 - h] = column2
    sum_feature = 0
    for r in range(2 * selectNum * claN):
        sum_feature += feature[0, r]
    for t in range(2 * selectNum * claN):
        feature[0, t] = np.log(feature[0, t] / sum_feature)
    return feature


def feature_extraction(input_data, input_label, input_freq_low, input_freq_high, sampleRate):
    trials = input_data.shape[0]
    channels = input_data.shape[1]
    samples = input_data.shape[2]
    filter_data = np.zeros((trials, channels, samples))
    classNu = len(np.unique(input_label))
    feature_train = np.zeros((trials, 2 * m * classNu * (len(input_freq_low))))
    projM_All = np.zeros((classNu * (len(input_freq_low)), channels, channels))
    for c in range(len(input_freq_low)):
        lower = input_freq_low[c]
        higher = input_freq_high[c]
        filter_tmp = np.zeros((channels, samples))
        for d in range(trials):
            filter_tmp = filter_filter(input_data[d, :, :], lower, higher, sampleRate)
            filter_data[d, :, :] = filter_tmp
        projM = csp_filter(filter_data, input_label)
        projM_All[c * classNu:(c + 1) * classNu, :, :] = projM
        feature_m = np.zeros((trials, 2 * m * classNu))
        for o in range(trials):
            feature_m[o, :] = csp_feature(projM, filter_data[o, :, :], m)
        feature_train[:, c * 2 * m * classNu:(c + 1) * 2 * m * classNu] = feature_m
    return feature_train, projM_All, classNu

train_matrix, proj, classNum = feature_extraction(data, label, freq_low, freq_high, samplerate)
np.save(r'F:\f.npy', train_matrix)
