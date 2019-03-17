"""
extract alignments voices.
"""

import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint

import numpy

import os
import pyworld
import matplotlib.pyplot as plot
import librosa.display

from become_yukarin.acoustic_converter import AcousticConverter
from become_yukarin.config.config import create_from_json as create_config
from become_yukarin.data_struct import AcousticFeature
from become_yukarin.dataset.dataset import AcousticFeatureLoadProcess
from become_yukarin.dataset.dataset import AcousticFeatureProcess
from become_yukarin.dataset.dataset import AcousticFeatureSaveProcess
from become_yukarin.dataset.dataset import WaveFileLoadProcess
from become_yukarin.dataset.utility import MelCepstrumAligner
from become_yukarin.param import AcousticFeatureParam
from become_yukarin.param import VoiceParam

base_voice_param = VoiceParam()
base_acoustic_feature_param = AcousticFeatureParam()

parser = argparse.ArgumentParser()
parser.add_argument('--input1_directory', '-i1', type=Path)
parser.add_argument('--input2_directory', '-i2', type=Path)
parser.add_argument('--output1_directory', '-o1', type=Path)
parser.add_argument('--output2_directory', '-o2', type=Path)
parser.add_argument('--pre_converter1_config', type=Path)
parser.add_argument('--pre_converter1_model', type=Path)
parser.add_argument('--sample_rate', type=int, default=base_voice_param.sample_rate)
parser.add_argument('--top_db', type=float, default=base_voice_param.top_db)
parser.add_argument('--pad_second', type=float, default=base_voice_param.pad_second)
parser.add_argument('--frame_period', type=int, default=base_acoustic_feature_param.frame_period)
parser.add_argument('--order', type=int, default=base_acoustic_feature_param.order)
parser.add_argument('--alpha', type=float, default=base_acoustic_feature_param.alpha)
parser.add_argument('--f0_estimating_method', type=str, default=base_acoustic_feature_param.f0_estimating_method)
parser.add_argument('--f0_floor1', type=float, default=71)
parser.add_argument('--f0_ceil1', type=float, default=800)
parser.add_argument('--f0_floor2', type=float, default=71)
parser.add_argument('--f0_ceil2', type=float, default=800)
parser.add_argument('--ignore_feature', nargs='+', default=['spectrogram', 'aperiodicity'])
parser.add_argument('--disable_alignment', action='store_true')
parser.add_argument('--enable_overwrite', action='store_true')
arguments = parser.parse_args()

pprint(dir(arguments))

def adjustment_len(arry1, arry2):
    #arry1 = replace_np_array(array1)
    #arry2 = replace_np_array(array2)

    o1_l = arry1.size
    o2_l = arry2.size
    if o1_l > o2_l:
        ex_l = o1_l - o2_l
        for i in range(ex_l):
            arry2 = numpy.append(arry2, 0.0)

    elif o2_l > o1_l:
        ex_l = o2_l - o1_l
        for i in range(ex_l):
            arry1 = numpy.append(arry1 , 0.0)

    return arry1, arry2


def replace_np_array(in_array):
    out = numpy.empty(0)
    for i in range(len(in_array)):
        out = numpy.append(out, in_array[i][0])

    return out


def plot_acoustic_feature(wave1, wave2, f1, f2, rate, write_path1):

    row_wave1, row_wave2 = adjustment_len(wave1, wave2)
    f1_f0, f2_f0 = adjustment_len(replace_np_array(f1.f0), replace_np_array(f2.f0))
    f1_spectrogram, f2_spectrogram = adjustment_len(replace_np_array(f1.spectrogram), replace_np_array(f2.spectrogram))
    f1_aperiodicity, f2_aperiodicity = adjustment_len(replace_np_array(f1.aperiodicity), replace_np_array(f2.aperiodicity))
    f1_voiced, f2_voiced = adjustment_len(replace_np_array(f1.voiced), replace_np_array(f2.voiced))

    plot.figure(figsize=(26, 13))
    plot.subplot(4,3,1)
    plot.title('speaker voice')
    librosa.display.waveplot(row_wave1, sr=rate)
    plot.xlabel('')

    plot.subplot(4,3,2)
    plot.title('f0')
    librosa.display.waveplot(f1_f0, sr=1, x_axis='none')

    plot.subplot(4,3,3)
    plot.title('aperiodicity')
    librosa.display.waveplot(f1_aperiodicity, sr=1, x_axis='none')

    plot.subplot(4,3,4)
    plot.title('voiced flag')
    librosa.display.waveplot(f1_voiced, sr=1, x_axis='none')

    plot.subplot(4,3,5)
    plot.title('spectral envelope')
    librosa.display.waveplot(f1_spectrogram, sr=1, x_axis='none')

    plot.subplot(4,3,6)
    plot.title('spectrum')
    plot.specgram(row_wave1,Fs = rate)

    plot.subplot(4,3,7)
    plot.title('target voice')
    librosa.display.waveplot(row_wave2, sr=rate)
    plot.xlabel('')

    plot.subplot(4,3,8)
    plot.title('f0')
    librosa.display.waveplot(f2_f0, sr=1, x_axis='none')

    plot.subplot(4,3,9)
    plot.title('aperiodicity')
    librosa.display.waveplot(f2_aperiodicity, sr=1, x_axis='none')

    plot.subplot(4,3,10)
    plot.title('voiced flag')
    librosa.display.waveplot(f2_voiced, sr=1, x_axis='none')

    plot.subplot(4,3,11)
    plot.title('spectral envelope')
    librosa.display.waveplot(f2_spectrogram, sr=1, x_axis='none')

    plot.subplot(4,3,12)
    plot.title('spectrum')
    plot.specgram(row_wave2,Fs = rate)

    plot.savefig(str(write_path1))
    plot.close()



def plot_acoustic_feature_for_check(wave1, wave2, f1, f2, rate, write_path1):

    row_wave1, row_wave2 = adjustment_len(wave1, wave2)
    f1_f0, f2_f0 = adjustment_len(replace_np_array(f1.f0), replace_np_array(f2.f0))

    plot.figure(figsize=(26, 13))
    plot.subplot(2,2,1)
    plot.title('origin')
    librosa.display.waveplot(row_wave1, sr=rate)

    plot.subplot(2,2,2)
    plot.title('f0')
    librosa.display.waveplot(f1_f0, sr=rate)

    plot.subplot(2,2,3)
    plot.title('origin')
    librosa.display.waveplot(row_wave2, sr=rate)

    plot.subplot(2,2,4)
    plot.title('f0')
    librosa.display.waveplot(f2_f0, sr=rate)

    plot.savefig(str(write_path1))
    plot.close()


pre_convert = arguments.pre_converter1_config is not None
if pre_convert:
    config = create_config(arguments.pre_converter1_config)
    pre_converter1 = AcousticConverter(config, arguments.pre_converter1_model)
else:
    pre_converter1 = None


def generate_feature(path1, path2):
    out1 = Path(arguments.output1_directory, path1.stem + '.npy')
    out2 = Path(arguments.output2_directory, path2.stem + '.npy')
    if out1.exists() and out2.exists() and not arguments.enable_overwrite:
        return

    # load wave and padding
    wave_file_load_process = WaveFileLoadProcess(
        sample_rate=arguments.sample_rate,
        top_db=arguments.top_db,
        pad_second=arguments.pad_second,
    )
    wave1 = wave_file_load_process(path1, test=True)
    wave2 = wave_file_load_process(path2, test=True)

    # make acoustic feature
    acoustic_feature_process1 = AcousticFeatureProcess(
        frame_period=arguments.frame_period,
        order=arguments.order,
        alpha=arguments.alpha,
        f0_estimating_method=arguments.f0_estimating_method,
        f0_floor=arguments.f0_floor1,
        f0_ceil=arguments.f0_ceil1,
    )
    acoustic_feature_process2 = AcousticFeatureProcess(
        frame_period=arguments.frame_period,
        order=arguments.order,
        alpha=arguments.alpha,
        f0_estimating_method=arguments.f0_estimating_method,
        f0_floor=arguments.f0_floor2,
        f0_ceil=arguments.f0_ceil2,
    )
    f1 = acoustic_feature_process1(wave1, test=True).astype_only_float(numpy.float32)
    f2 = acoustic_feature_process2(wave2, test=True).astype_only_float(numpy.float32)

    # pre convert
    if pre_convert:
        f1_ref = pre_converter1.convert_to_feature(f1)
    else:
        f1_ref = f1

    # alignment
    if not arguments.disable_alignment:
        aligner = MelCepstrumAligner(f1_ref.mfcc, f2.mfcc)

        f0_1, f0_2 = aligner.align(f1.f0, f2.f0)
        spectrogram_1, spectrogram_2 = aligner.align(f1.spectrogram, f2.spectrogram)
        aperiodicity_1, aperiodicity_2 = aligner.align(f1.aperiodicity, f2.aperiodicity)
        mfcc_1, mfcc_2 = aligner.align(f1.mfcc, f2.mfcc)
        voiced_1, voiced_2 = aligner.align(f1.voiced, f2.voiced)

        f1 = AcousticFeature(
            f0=f0_1,
            spectrogram=spectrogram_1,
            aperiodicity=aperiodicity_1,
            mfcc=mfcc_1,
            voiced=voiced_1,
        )
        f2 = AcousticFeature(
            f0=f0_2,
            spectrogram=spectrogram_2,
            aperiodicity=aperiodicity_2,
            mfcc=mfcc_2,
            voiced=voiced_2,
        )

        f1.validate()
        f2.validate()

        wave1 = pyworld.synthesize(
            f0=numpy.array(f0_1.ravel(), dtype='float64'),
            spectrogram=numpy.array(spectrogram_1, dtype='float64'),
            aperiodicity=numpy.array(aperiodicity_1, dtype='float64'),
            fs=arguments.sample_rate,
            frame_period=arguments.frame_period
        )

        wave2 = pyworld.synthesize(
            f0=numpy.array(f0_2.ravel(), dtype='float64'),
            spectrogram=numpy.array(spectrogram_2, dtype='float64'),
            aperiodicity=numpy.array(aperiodicity_2, dtype='float64'),
            fs=arguments.sample_rate,
            frame_period=arguments.frame_period
        )

        out_p1 = Path(str(arguments.output1_directory) + "_ajst_plot", path1.stem)
        plot_acoustic_feature(wave1, wave2, f1, f2, arguments.sample_rate, out_p1)

        librosa.output.write_wav(str(out_p1) + '_my.wav', wave1, arguments.sample_rate, norm=True)
        librosa.output.write_wav(str(out_p1) + '_yuka.wav', wave2, arguments.sample_rate, norm=True)

    # save
    acoustic_feature_save_process = AcousticFeatureSaveProcess(validate=True, ignore=arguments.ignore_feature)
    acoustic_feature_save_process({'path': out1, 'feature': f1})
    print('saved!', out1)

    acoustic_feature_save_process({'path': out2, 'feature': f2})
    print('saved!', out2)


def generate_mean_var(path_directory: Path):
    path_mean = Path(path_directory, 'mean.npy')
    path_var = Path(path_directory, 'var.npy')
    if path_mean.exists():
        path_mean.unlink()
    if path_var.exists():
        path_var.unlink()

    acoustic_feature_load_process = AcousticFeatureLoadProcess(validate=False)
    acoustic_feature_save_process = AcousticFeatureSaveProcess(validate=False)

    f0_list = []
    spectrogram_list = []
    aperiodicity_list = []
    mfcc_list = []
    for path in path_directory.glob('*.npy'):
        feature = acoustic_feature_load_process(path)
        f0_list.append(feature.f0[feature.voiced])  # remove unvoiced
        spectrogram_list.append(feature.spectrogram)
        aperiodicity_list.append(feature.aperiodicity)
        mfcc_list.append(feature.mfcc)

    def concatenate(arr_list):
        try:
            arr_list = numpy.concatenate(arr_list)
        except:
            pass
        return arr_list

    f0_list = concatenate(f0_list)
    spectrogram_list = concatenate(spectrogram_list)
    aperiodicity_list = concatenate(aperiodicity_list)
    mfcc_list = concatenate(mfcc_list)

    mean = AcousticFeature(
        f0=numpy.mean(f0_list, axis=0, keepdims=True),
        spectrogram=numpy.mean(spectrogram_list, axis=0, keepdims=True),
        aperiodicity=numpy.mean(aperiodicity_list, axis=0, keepdims=True),
        mfcc=numpy.mean(mfcc_list, axis=0, keepdims=True),
        voiced=numpy.nan,
    )
    var = AcousticFeature(
        f0=numpy.var(f0_list, axis=0, keepdims=True),
        spectrogram=numpy.var(spectrogram_list, axis=0, keepdims=True),
        aperiodicity=numpy.var(aperiodicity_list, axis=0, keepdims=True),
        mfcc=numpy.var(mfcc_list, axis=0, keepdims=True),
        voiced=numpy.nan,
    )

    acoustic_feature_save_process({'path': path_mean, 'feature': mean})
    acoustic_feature_save_process({'path': path_var, 'feature': var})


def main():
    paths1 = list(sorted(arguments.input1_directory.glob('*.wav')))
    paths2 = list(sorted(arguments.input2_directory.glob('*.wav')))
    assert len(paths1) == len(paths2)

    arguments.output1_directory.mkdir(exist_ok=True)
    arguments.output2_directory.mkdir(exist_ok=True)

    #pool = multiprocessing.Pool()
    #pool.starmap(generate_feature, zip(paths1, paths2), chunksize=16)

    for i in range(len(paths1)):
        generate_feature(paths1[i], paths2[i])


    generate_mean_var(arguments.output1_directory)
    generate_mean_var(arguments.output2_directory)


if __name__ == '__main__':
    main()
