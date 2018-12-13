# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:50:58 2018

@author: Administrator
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class bgn_analysis(object):
    def __init__(self,file_name,noise_name,offset,duration):
        self.__filename=file_name
        self.__noisename=noise_name
        self.__offset=offset
        self.__duration=duration
    
    def __wav_load(self):
        self.__wav_data,self.__sr=librosa.load(self.__filename,sr=None,mono=False,offset=self.__offset,duration=self.__duration)
    
    def __wav_analysis_init(self):
        self.__wav_load()
        self.__winlen=4096
        self.__hoplen=1024
        self.__eps=10**(-8)
    
    def __stft_analysis(self):
        self.__wav_analysis_init()
        self.__stft_amp_left_channel,self.__stft_phase_left_channel=librosa.magphase(librosa.stft(self.__wav_data[0,:],n_fft=self.__winlen,hop_length=self.__hoplen))
        self.__stft_amp_right_channel,self.__stft_phase_right_channel=librosa.magphase(librosa.stft(self.__wav_data[1,:],n_fft=self.__winlen,hop_length=self.__hoplen))
        self.__stft_amp_dB_left_channel=20*np.log10(self.__stft_amp_left_channel/self.__winlen+self.__eps)
        self.__stft_amp_dB_right_channel=20*np.log10(self.__stft_amp_right_channel/self.__winlen+self.__eps)           

    def Spectrum_Display(self):
        self.__stft_analysis()
        self.__frequency=np.arange(len(self.__stft_amp_left_channel))*self.__sr/self.__winlen
        plt.semilogx(self.__frequency,np.max(self.__stft_amp_dB_left_channel,axis=1))
        plt.semilogx(self.__frequency,np.max(self.__stft_amp_dB_right_channel,axis=1))
        plt.legend(labels=["Left Channel","Right Channel"],loc="best")
        plt.title(self.__noisename+" FR")
        plt.ylabel("Amplitude (dB)")
        plt.grid(1)

    def RMSE_Display(self):
        self.__wav_analysis_init()
        self.__rms_left_channel=20*np.log10(librosa.feature.rmse(y=self.__wav_data[0,:],frame_length=4800,hop_length=1200)[0]+self.__eps)
        self.__rms_right_channel=20*np.log10(librosa.feature.rmse(y=self.__wav_data[1,:],frame_length=4800,hop_length=1200)[0]+self.__eps)       
        self.__time=librosa.frames_to_time(np.arange(len(self.__rms_left_channel)),sr=self.__sr,hop_length=1200,n_fft=4800)
        plt.plot(self.__time,self.__rms_left_channel)
        plt.plot(self.__time,self.__rms_right_channel)
        plt.xlim(0)
        plt.legend(labels=["Left Channel","Right Channel"],loc="best")
        plt.ylabel("RMS (dB)")
        plt.grid(1)
        plt.title(self.__noisename+" RMS")

    def Left_Spectrogram_Display(self):
        self.__stft_analysis()        
        librosa.display.specshow(librosa.amplitude_to_db(self.__stft_amp_left_channel,ref=1.0),x_axis='time',y_axis='log',sr=self.__sr,hop_length=self.__hoplen)
        plt.colorbar()
        plt.title(self.__noisename+" Left Channel Spectrogram")

    def Right_Spectrogram_Display(self):
        self.__stft_analysis()        
        librosa.display.specshow(librosa.amplitude_to_db(self.__stft_amp_right_channel,ref=1.0),x_axis='time',y_axis='log',sr=self.__sr,hop_length=self.__hoplen)
        plt.colorbar()
        plt.title(self.__noisename+" Right Channel Spectrogram")
        
if __name__=='__main__':
    
    pub_noise=bgn_analysis(r"D:\HEADAutoEQ_3.1.200\HEAD acoustics\HEADAutoEQ\Background Noise\Pub_bin.wav","Pub Noise",0,None)
    road_noise=bgn_analysis(r"D:\HEADAutoEQ_3.1.200\HEAD acoustics\HEADAutoEQ\Background Noise\Roadnoise_bin.wav","Road Noise",0,None)
    xroad_noise=bgn_analysis(r"D:\HEADAutoEQ_3.1.200\HEAD acoustics\HEADAutoEQ\Background Noise\Crossroadnoise_bin.wav","Crossroad Noise",0,None)
    train_noise=bgn_analysis(r"D:\HEADAutoEQ_3.1.200\HEAD acoustics\HEADAutoEQ\Background Noise\TrainStation_bin.wav","Train Noise",0,None)
    car_noise=bgn_analysis(r"D:\HEADAutoEQ_3.1.200\HEAD acoustics\HEADAutoEQ\Background Noise\FullSizeCar_130_bin.wav","Car Noise",0,None)
    cafe_noise=bgn_analysis(r"D:\HEADAutoEQ_3.1.200\HEAD acoustics\HEADAutoEQ\Background Noise\Cafeteria_bin.wav","Cafe Noise",0,None)
    mensa_noise=bgn_analysis(r"D:\HEADAutoEQ_3.1.200\HEAD acoustics\HEADAutoEQ\Background Noise\Mensa_63dB.wav","Mensa Noise",0,None)
    callcenter_noise=bgn_analysis(r"D:\HEADAutoEQ_3.1.200\HEAD acoustics\HEADAutoEQ\Background Noise\Callcenter1_bin.wav","CallCenter Noise",0,None)

    '''
    plt.figure(1)
    plt.subplot(421)
    pub_noise.Spectrum_Display()
    plt.subplot(422)
    road_noise.Spectrum_Display()
    plt.subplot(423)
    xroad_noise.Spectrum_Display()
    plt.subplot(424)
    train_noise.Spectrum_Display()
    plt.subplot(425)
    car_noise.Spectrum_Display()
    plt.subplot(426)
    cafe_noise.Spectrum_Display()
    plt.subplot(427)
    mensa_noise.Spectrum_Display()
    plt.xlabel("Frequency (Hz)")
    plt.subplot(428)
    callcenter_noise.Spectrum_Display()
    plt.xlabel("Frequency (Hz)")

    plt.figure(2)
    plt.subplot(421)
    pub_noise.RMSE_Display()
    plt.subplot(422)
    road_noise.RMSE_Display()
    plt.subplot(423)
    xroad_noise.RMSE_Display()
    plt.subplot(424)
    train_noise.RMSE_Display()
    plt.subplot(425)
    car_noise.RMSE_Display()
    plt.subplot(426)
    cafe_noise.RMSE_Display()
    plt.subplot(427)
    mensa_noise.RMSE_Display()
    plt.xlabel("Time (s)")
    plt.subplot(428)
    callcenter_noise.RMSE_Display()
    plt.xlabel("Time (s)")

    plt.figure(3)
    plt.subplot(421)
    pub_noise.Left_Spectrogram_Display()
    plt.subplot(422)
    road_noise.Left_Spectrogram_Display()
    plt.subplot(423)
    xroad_noise.Left_Spectrogram_Display()
    plt.subplot(424)
    train_noise.Left_Spectrogram_Display()
    plt.subplot(425)
    car_noise.Left_Spectrogram_Display()
    plt.subplot(426)
    cafe_noise.Left_Spectrogram_Display()
    plt.subplot(427)
    mensa_noise.Left_Spectrogram_Display()
    plt.subplot(428)
    callcenter_noise.Left_Spectrogram_Display()

    plt.figure(4)
    plt.subplot(421)
    pub_noise.Right_Spectrogram_Display()
    plt.subplot(422)
    road_noise.Right_Spectrogram_Display()
    plt.subplot(423)
    xroad_noise.Right_Spectrogram_Display()
    plt.subplot(424)
    train_noise.Right_Spectrogram_Display()
    plt.subplot(425)
    car_noise.Right_Spectrogram_Display()
    plt.subplot(426)
    cafe_noise.Right_Spectrogram_Display()
    plt.subplot(427)
    mensa_noise.Right_Spectrogram_Display()
    plt.subplot(428)
    callcenter_noise.Right_Spectrogram_Display()
    '''


    plt.show()



